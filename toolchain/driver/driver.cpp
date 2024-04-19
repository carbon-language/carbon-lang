// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/driver.h"

#include <algorithm>
#include <memory>
#include <optional>

#include "common/command_line.h"
#include "common/vlog.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/Host.h"
#include "toolchain/base/value_store.h"
#include "toolchain/check/check.h"
#include "toolchain/codegen/codegen.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/sorting_diagnostic_consumer.h"
#include "toolchain/lex/lex.h"
#include "toolchain/lower/lower.h"
#include "toolchain/parse/parse.h"
#include "toolchain/sem_ir/formatter.h"
#include "toolchain/sem_ir/inst_namer.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon {

struct Driver::CompileOptions {
  static constexpr CommandLine::CommandInfo Info = {
      .name = "compile",
      .help = R"""(
Compile Carbon source code.

This subcommand runs the Carbon compiler over input source code, checking it for
errors and producing the requested output.

Error messages are written to the standard error stream.

Different phases of the compiler can be selected to run, and intermediate state
can be written to standard output as these phases progress.
)""",
  };

  enum class Phase : int8_t {
    Lex,
    Parse,
    Check,
    Lower,
    CodeGen,
  };

  friend auto operator<<(llvm::raw_ostream& out, Phase phase)
      -> llvm::raw_ostream& {
    switch (phase) {
      case Phase::Lex:
        out << "lex";
        break;
      case Phase::Parse:
        out << "parse";
        break;
      case Phase::Check:
        out << "check";
        break;
      case Phase::Lower:
        out << "lower";
        break;
      case Phase::CodeGen:
        out << "codegen";
        break;
    }
    return out;
  }

  void Build(CommandLine::CommandBuilder& b) {
    b.AddStringPositionalArg(
        {
            .name = "FILE",
            .help = R"""(
The input Carbon source file to compile.
)""",
        },
        [&](auto& arg_b) {
          arg_b.Required(true);
          arg_b.Append(&input_filenames);
        });

    b.AddOneOfOption(
        {
            .name = "phase",
            .help = R"""(
Selects the compilation phase to run. These phases are always run in sequence,
so every phase before the one selected will also be run. The default is to
compile to machine code.
)""",
        },
        [&](auto& arg_b) {
          arg_b.SetOneOf(
              {
                  arg_b.OneOfValue("lex", Phase::Lex),
                  arg_b.OneOfValue("parse", Phase::Parse),
                  arg_b.OneOfValue("check", Phase::Check),
                  arg_b.OneOfValue("lower", Phase::Lower),
                  arg_b.OneOfValue("codegen", Phase::CodeGen).Default(true),
              },
              &phase);
        });

    // TODO: Rearrange the code setting this option and two related ones to
    // allow them to reference each other instead of hard-coding their names.
    b.AddStringOption(
        {
            .name = "output",
            .value_name = "FILE",
            .help = R"""(
The output filename for codegen.

When this is a file name, either textual assembly or a binary object will be
written to it based on the flag `--asm-output`. The default is to write a binary
object file.

Passing `--output=-` will write the output to stdout. In that case, the flag
`--asm-output` is ignored and the output defaults to textual assembly. Binary
object output can be forced by enabling `--force-obj-output`.
)""",
        },
        [&](auto& arg_b) { arg_b.Set(&output_filename); });

    b.AddStringOption(
        {
            .name = "target",
            .help = R"""(
Select a target platform. Uses the LLVM target syntax. Also known as a "triple"
for historical reasons.

This corresponds to the `target` flag to Clang and accepts the same strings
documented there:
https://clang.llvm.org/docs/CrossCompilation.html#target-triple
)""",
        },
        [&](auto& arg_b) {
          arg_b.Default(host);
          arg_b.Set(&target);
        });

    b.AddFlag(
        {
            .name = "asm-output",
            .help = R"""(
Write textual assembly rather than a binary object file to the code generation
output.

This flag only applies when writing to a file. When writing to stdout, the
default is textual assembly and this flag is ignored.
)""",
        },
        [&](auto& arg_b) { arg_b.Set(&asm_output); });

    b.AddFlag(
        {
            .name = "force-obj-output",
            .help = R"""(
Force binary object output, even with `--output=-`.

When `--output=-` is set, the default is textual assembly; this forces printing
of a binary object file instead. Ignored for other `--output` values.
)""",
        },
        [&](auto& arg_b) { arg_b.Set(&force_obj_output); });

    b.AddFlag(
        {
            .name = "stream-errors",
            .help = R"""(
Stream error messages to stderr as they are generated rather than sorting them
and displaying them in source order.
)""",
        },
        [&](auto& arg_b) { arg_b.Set(&stream_errors); });

    b.AddFlag(
        {
            .name = "dump-shared-values",
            .help = R"""(
Dumps shared values. These aren't owned by any particular file or phase.
)""",
        },
        [&](auto& arg_b) { arg_b.Set(&dump_shared_values); });
    b.AddFlag(
        {
            .name = "dump-tokens",
            .help = R"""(
Dump the tokens to stdout when lexed.
)""",
        },
        [&](auto& arg_b) { arg_b.Set(&dump_tokens); });
    b.AddFlag(
        {
            .name = "dump-parse-tree",
            .help = R"""(
Dump the parse tree to stdout when parsed.
)""",
        },
        [&](auto& arg_b) { arg_b.Set(&dump_parse_tree); });
    b.AddFlag(
        {
            .name = "preorder-parse-tree",
            .help = R"""(
When dumping the parse tree, reorder it so that it is in preorder rather than
postorder.
)""",
        },
        [&](auto& arg_b) { arg_b.Set(&preorder_parse_tree); });
    b.AddFlag(
        {
            .name = "dump-raw-sem-ir",
            .help = R"""(
Dump the raw JSON structure of SemIR to stdout when built.
)""",
        },
        [&](auto& arg_b) { arg_b.Set(&dump_raw_sem_ir); });
    b.AddFlag(
        {
            .name = "dump-sem-ir",
            .help = R"""(
Dump the SemIR to stdout when built.
)""",
        },
        [&](auto& arg_b) { arg_b.Set(&dump_sem_ir); });
    b.AddFlag(
        {
            .name = "builtin-sem-ir",
            .help = R"""(
Include the SemIR for builtins when dumping it.
)""",
        },
        [&](auto& arg_b) { arg_b.Set(&builtin_sem_ir); });
    b.AddFlag(
        {
            .name = "dump-llvm-ir",
            .help = R"""(
Dump the LLVM IR to stdout after lowering.
)""",
        },
        [&](auto& arg_b) { arg_b.Set(&dump_llvm_ir); });
    b.AddFlag(
        {
            .name = "dump-asm",
            .help = R"""(
Dump the generated assembly to stdout after codegen.
)""",
        },
        [&](auto& arg_b) { arg_b.Set(&dump_asm); });
    b.AddFlag(
        {
            .name = "prelude-import",
            .help = R"""(
Whether to use the implicit prelude import. Enabled by default.
)""",
        },
        [&](auto& arg_b) {
          arg_b.Default(true);
          arg_b.Set(&prelude_import);
        });
    b.AddStringOption(
        {
            .name = "exclude-dump-file-prefix",
            .value_name = "PREFIX",
            .help = R"""(
Excludes files with the given prefix from dumps.
)""",
        },
        [&](auto& arg_b) { arg_b.Set(&exclude_dump_file_prefix); });
  }

  Phase phase;

  std::string host = llvm::sys::getDefaultTargetTriple();
  llvm::StringRef target;

  llvm::StringRef output_filename;
  llvm::SmallVector<llvm::StringRef> input_filenames;

  bool asm_output = false;
  bool force_obj_output = false;
  bool dump_shared_values = false;
  bool dump_tokens = false;
  bool dump_parse_tree = false;
  bool dump_raw_sem_ir = false;
  bool dump_sem_ir = false;
  bool dump_llvm_ir = false;
  bool dump_asm = false;
  bool stream_errors = false;
  bool preorder_parse_tree = false;
  bool builtin_sem_ir = false;
  bool prelude_import = false;

  llvm::StringRef exclude_dump_file_prefix;
};

struct Driver::Options {
  static constexpr CommandLine::CommandInfo Info = {
      .name = "carbon",
      // TODO: Set up more detailed version information and use that here.
      .version = R"""(
Carbon Language toolchain -- version 0.0.0
)""",
      .help = R"""(
This is the unified Carbon Language toolchain driver. Its subcommands provide
all of the core behavior of the toolchain, including compilation, linking, and
developer tools. Each of these has its own subcommand, and you can pass a
specific subcommand to the `help` subcommand to get details about its usage.
)""",
      .help_epilogue = R"""(
For questions, issues, or bug reports, please use our GitHub project:

  https://github.com/carbon-language/carbon-lang
)""",
  };

  enum class Subcommand : int8_t {
    Compile,
  };

  void Build(CommandLine::CommandBuilder& b) {
    b.AddFlag(
        {
            .name = "verbose",
            .short_name = "v",
            .help = "Enable verbose logging to the stderr stream.",
        },
        [&](CommandLine::FlagBuilder& arg_b) { arg_b.Set(&verbose); });

    b.AddSubcommand(CompileOptions::Info,
                    [&](CommandLine::CommandBuilder& sub_b) {
                      compile_options.Build(sub_b);
                      sub_b.Do([&] { subcommand = Subcommand::Compile; });
                    });

    b.RequiresSubcommand();
  }

  bool verbose;
  Subcommand subcommand;

  CompileOptions compile_options;
};

auto Driver::ParseArgs(llvm::ArrayRef<llvm::StringRef> args, Options& options)
    -> CommandLine::ParseResult {
  return CommandLine::Parse(
      args, output_stream_, error_stream_, Options::Info,
      [&](CommandLine::CommandBuilder& b) { options.Build(b); });
}

auto Driver::RunCommand(llvm::ArrayRef<llvm::StringRef> args) -> RunResult {
  Options options;
  CommandLine::ParseResult result = ParseArgs(args, options);
  if (result == CommandLine::ParseResult::Error) {
    return {.success = false};
  } else if (result == CommandLine::ParseResult::MetaSuccess) {
    return {.success = true};
  }

  if (options.verbose) {
    // Note this implies streamed output in order to interleave.
    vlog_stream_ = &error_stream_;
  }

  switch (options.subcommand) {
    case Options::Subcommand::Compile:
      return Compile(options.compile_options);
  }
  llvm_unreachable("All subcommands handled!");
}

auto Driver::ValidateCompileOptions(const CompileOptions& options) const
    -> bool {
  using Phase = CompileOptions::Phase;
  switch (options.phase) {
    case Phase::Lex:
      if (options.dump_parse_tree) {
        error_stream_ << "ERROR: Requested dumping the parse tree but compile "
                         "phase is limited to '"
                      << options.phase << "'.\n";
        return false;
      }
      [[fallthrough]];
    case Phase::Parse:
      if (options.dump_sem_ir) {
        error_stream_ << "ERROR: Requested dumping the SemIR but compile phase "
                         "is limited to '"
                      << options.phase << "'.\n";
        return false;
      }
      [[fallthrough]];
    case Phase::Check:
      if (options.dump_llvm_ir) {
        error_stream_ << "ERROR: Requested dumping the LLVM IR but compile "
                         "phase is limited to '"
                      << options.phase << "'.\n";
        return false;
      }
      [[fallthrough]];
    case Phase::Lower:
    case Phase::CodeGen:
      // Everything can be dumped in these phases.
      break;
  }
  return true;
}

// Ties together information for a file being compiled.
class Driver::CompilationUnit {
 public:
  explicit CompilationUnit(Driver* driver, const CompileOptions& options,
                           DiagnosticConsumer* consumer,
                           llvm::StringRef input_filename)
      : driver_(driver),
        options_(options),
        input_filename_(input_filename),
        vlog_stream_(driver_->vlog_stream_) {
    if (vlog_stream_ != nullptr || options_.stream_errors) {
      consumer_ = consumer;
    } else {
      sorting_consumer_ = SortingDiagnosticConsumer(*consumer);
      consumer_ = &*sorting_consumer_;
    }
  }

  // Loads source and lexes it. Returns true on success.
  auto RunLex() -> void {
    LogCall("SourceBuffer::MakeFromFile", [&] {
      if (input_filename_ == "-") {
        source_ = SourceBuffer::MakeFromStdin(*consumer_);
      } else {
        source_ = SourceBuffer::MakeFromFile(driver_->fs_, input_filename_,
                                             *consumer_);
      }
    });
    if (!source_) {
      success_ = false;
      return;
    }
    CARBON_VLOG() << "*** SourceBuffer ***\n```\n"
                  << source_->text() << "\n```\n";

    LogCall("Lex::Lex",
            [&] { tokens_ = Lex::Lex(value_stores_, *source_, *consumer_); });
    if (options_.dump_tokens && IncludeInDumps()) {
      consumer_->Flush();
      driver_->output_stream_ << tokens_;
    }
    CARBON_VLOG() << "*** Lex::TokenizedBuffer ***\n" << tokens_;
    if (tokens_->has_errors()) {
      success_ = false;
    }
  }

  // Parses tokens. Returns true on success.
  auto RunParse() -> void {
    CARBON_CHECK(tokens_);

    LogCall("Parse::Parse", [&] {
      parse_tree_ = Parse::Parse(*tokens_, *consumer_, vlog_stream_);
    });
    if (options_.dump_parse_tree && IncludeInDumps()) {
      consumer_->Flush();
      parse_tree_->Print(driver_->output_stream_, options_.preorder_parse_tree);
    }
    CARBON_VLOG() << "*** Parse::Tree ***\n" << parse_tree_;
    if (parse_tree_->has_errors()) {
      success_ = false;
    }
  }

  // Returns information needed to check this unit.
  auto GetCheckUnit() -> Check::Unit {
    CARBON_CHECK(parse_tree_);
    return {.value_stores = &value_stores_,
            .tokens = &*tokens_,
            .parse_tree = &*parse_tree_,
            .consumer = consumer_,
            .sem_ir = &sem_ir_};
  }

  // Runs post-check logic. Returns true if checking succeeded for the IR.
  auto PostCheck() -> void {
    CARBON_CHECK(sem_ir_);

    // We've finished all steps that can produce diagnostics. Emit the
    // diagnostics now, so that the developer sees them sooner and doesn't need
    // to wait for code generation.
    consumer_->Flush();

    CARBON_VLOG() << "*** Raw SemIR::File ***\n" << *sem_ir_ << "\n";
    if (options_.dump_raw_sem_ir && IncludeInDumps()) {
      sem_ir_->Print(driver_->output_stream_, options_.builtin_sem_ir);
      if (options_.dump_sem_ir) {
        driver_->output_stream_ << "\n";
      }
    }

    if (vlog_stream_) {
      CARBON_VLOG() << "*** SemIR::File ***\n";
      SemIR::FormatFile(*tokens_, *parse_tree_, *sem_ir_, *vlog_stream_);
    }
    if (options_.dump_sem_ir && IncludeInDumps()) {
      SemIR::FormatFile(*tokens_, *parse_tree_, *sem_ir_,
                        driver_->output_stream_);
    }
    if (sem_ir_->has_errors()) {
      success_ = false;
    }
  }

  // Lower SemIR to LLVM IR.
  auto RunLower() -> void {
    CARBON_CHECK(sem_ir_);

    LogCall("Lower::LowerToLLVM", [&] {
      llvm_context_ = std::make_unique<llvm::LLVMContext>();
      // TODO: Consider disabling instruction naming by default if we're not
      // producing textual LLVM IR.
      SemIR::InstNamer inst_namer(*tokens_, *parse_tree_, *sem_ir_);
      module_ = Lower::LowerToLLVM(*llvm_context_, input_filename_, *sem_ir_,
                                   &inst_namer, vlog_stream_);
    });
    if (vlog_stream_) {
      CARBON_VLOG() << "*** llvm::Module ***\n";
      module_->print(*vlog_stream_, /*AAW=*/nullptr,
                     /*ShouldPreserveUseListOrder=*/false,
                     /*IsForDebug=*/true);
    }
    if (options_.dump_llvm_ir && IncludeInDumps()) {
      module_->print(driver_->output_stream_, /*AAW=*/nullptr,
                     /*ShouldPreserveUseListOrder=*/true);
    }
  }

  auto RunCodeGen() -> void {
    CARBON_CHECK(module_);
    LogCall("CodeGen", [&] { success_ = RunCodeGenHelper(); });
  }

  // Flushes output.
  auto Flush() -> void { consumer_->Flush(); }

  auto PrintSharedValues() const -> void {
    Yaml::Print(driver_->output_stream_,
                value_stores_.OutputYaml(input_filename_));
  }

  auto input_filename() -> llvm::StringRef { return input_filename_; }
  auto success() -> bool { return success_; }
  auto has_source() -> bool { return source_.has_value(); }

 private:
  // Do codegen. Returns true on success.
  auto RunCodeGenHelper() -> bool {
    std::optional<CodeGen> codegen =
        CodeGen::Make(*module_, options_.target, driver_->error_stream_);
    if (!codegen) {
      return false;
    }
    if (vlog_stream_) {
      CARBON_VLOG() << "*** Assembly ***\n";
      codegen->EmitAssembly(*vlog_stream_);
    }

    if (options_.output_filename == "-") {
      // TODO: the output file name, forcing object output, and requesting
      // textual assembly output are all somewhat linked flags. We should add
      // some validation that they are used correctly.
      if (options_.force_obj_output) {
        if (!codegen->EmitObject(driver_->output_stream_)) {
          return false;
        }
      } else {
        if (!codegen->EmitAssembly(driver_->output_stream_)) {
          return false;
        }
      }
    } else {
      llvm::SmallString<256> output_filename = options_.output_filename;
      if (output_filename.empty()) {
        if (!source_->is_regular_file()) {
          // Don't invent file names like `-.o` or `/dev/stdin.o`.
          driver_->error_stream_
              << "ERROR: Output file name must be specified for input '"
              << input_filename_ << "' that is not a regular file.\n";
          return false;
        }
        output_filename = input_filename_;
        llvm::sys::path::replace_extension(output_filename,
                                           options_.asm_output ? ".s" : ".o");
      } else {
        // TODO: Handle the case where multiple input files were specified
        // along with an output file name. That should either be an error or
        // should produce a single LLVM IR module containing all inputs.
        // Currently each unit overwrites the output from the previous one in
        // this case.
      }
      CARBON_VLOG() << "Writing output to: " << output_filename << "\n";

      std::error_code ec;
      llvm::raw_fd_ostream output_file(output_filename, ec,
                                       llvm::sys::fs::OF_None);
      if (ec) {
        driver_->error_stream_ << "ERROR: Could not open output file '"
                               << output_filename << "': " << ec.message()
                               << "\n";
        return false;
      }
      if (options_.asm_output) {
        if (!codegen->EmitAssembly(output_file)) {
          return false;
        }
      } else {
        if (!codegen->EmitObject(output_file)) {
          return false;
        }
      }
    }
    return true;
  }

  // Wraps a call with log statements to indicate start and end.
  auto LogCall(llvm::StringLiteral label, llvm::function_ref<void()> fn)
      -> void {
    CARBON_VLOG() << "*** " << label << ": " << input_filename_ << " ***\n";
    fn();
    CARBON_VLOG() << "*** " << label << " done ***\n";
  }

  // Returns true if the file can be dumped.
  auto IncludeInDumps() const -> bool {
    return options_.exclude_dump_file_prefix.empty() ||
           !input_filename_.starts_with(options_.exclude_dump_file_prefix);
  }

  Driver* driver_;
  SharedValueStores value_stores_;
  const CompileOptions& options_;
  std::string input_filename_;

  // Copied from driver_ for CARBON_VLOG.
  llvm::raw_pwrite_stream* vlog_stream_;

  // Diagnostics are sent to consumer_, with optional sorting.
  std::optional<SortingDiagnosticConsumer> sorting_consumer_;
  DiagnosticConsumer* consumer_;

  bool success_ = true;

  // These are initialized as steps are run.
  std::optional<SourceBuffer> source_;
  std::optional<Lex::TokenizedBuffer> tokens_;
  std::optional<Parse::Tree> parse_tree_;
  std::optional<SemIR::File> sem_ir_;
  std::unique_ptr<llvm::LLVMContext> llvm_context_;
  std::unique_ptr<llvm::Module> module_;
};

auto Driver::Compile(const CompileOptions& options) -> RunResult {
  if (!ValidateCompileOptions(options)) {
    return {.success = false};
  }

  // Prepare CompilationUnits before building scope exit handlers.
  StreamDiagnosticConsumer stream_consumer(error_stream_);
  llvm::SmallVector<std::unique_ptr<CompilationUnit>> units;
  units.reserve(options.prelude_import + options.input_filenames.size());

  // Directly insert the core package into the compilation units.
  // TODO: Should expand this into a more rich system to search for the core
  // package source code.
  if (options.prelude_import) {
    llvm::SmallString<256> prelude_file(data_dir_);
    llvm::sys::path::append(prelude_file, llvm::sys::path::Style::posix,
                            "core/prelude.carbon");
    units.push_back(std::make_unique<CompilationUnit>(
        this, options, &stream_consumer, prelude_file));
  }

  // Add the input source files.
  for (const auto& input_filename : options.input_filenames) {
    units.push_back(std::make_unique<CompilationUnit>(
        this, options, &stream_consumer, input_filename));
  }

  auto on_exit = llvm::make_scope_exit([&]() {
    // Shared values will always be printed after per-file printing.
    if (options.dump_shared_values) {
      for (const auto& unit : units) {
        unit->PrintSharedValues();
      }
    }

    // The diagnostics consumer must be flushed before compilation artifacts are
    // destructed, because diagnostics can refer to their state. This ensures
    // they're flushed in order of arguments, rather than order of destruction.
    for (auto& unit : units) {
      unit->Flush();
    }
    stream_consumer.Flush();
  });

  // Returns a RunResult object. Called whenever Compile returns.
  auto make_result = [&]() {
    RunResult result = {.success = true};
    for (const auto& unit : units) {
      result.success &= unit->success();
      result.per_file_success.push_back(
          {unit->input_filename().str(), unit->success()});
    }
    return result;
  };

  // Lex.
  for (auto& unit : units) {
    unit->RunLex();
  }
  if (options.phase == CompileOptions::Phase::Lex) {
    return make_result();
  }
  // Parse and check phases examine `has_source` because they want to proceed if
  // lex failed, but not if source doesn't exist. Later steps are skipped if
  // anything failed, so don't need this.

  // Parse.
  for (auto& unit : units) {
    if (unit->has_source()) {
      unit->RunParse();
    }
  }
  if (options.phase == CompileOptions::Phase::Parse) {
    return make_result();
  }

  // Check.
  SharedValueStores builtin_value_stores;
  auto builtins = Check::MakeBuiltins(builtin_value_stores);
  llvm::SmallVector<Check::Unit> check_units;
  for (auto& unit : units) {
    if (unit->has_source()) {
      check_units.push_back(unit->GetCheckUnit());
    }
  }
  CARBON_VLOG() << "*** Check::CheckParseTrees ***\n";
  Check::CheckParseTrees(builtins, llvm::MutableArrayRef(check_units),
                         options.prelude_import, vlog_stream_);
  CARBON_VLOG() << "*** Check::CheckParseTrees done ***\n";
  for (auto& unit : units) {
    if (unit->has_source()) {
      unit->PostCheck();
    }
  }
  if (options.phase == CompileOptions::Phase::Check) {
    return make_result();
  }

  // Unlike previous steps, errors block further progress.
  if (std::any_of(units.begin(), units.end(),
                  [&](const auto& unit) { return !unit->success(); })) {
    CARBON_VLOG() << "*** Stopping before lowering due to errors ***";
    return make_result();
  }

  // Lower.
  for (auto& unit : units) {
    unit->RunLower();
  }
  if (options.phase == CompileOptions::Phase::Lower) {
    return make_result();
  }
  CARBON_CHECK(options.phase == CompileOptions::Phase::CodeGen)
      << "CodeGen should be the last stage";

  // Codegen.
  for (auto& unit : units) {
    unit->RunCodeGen();
  }
  return make_result();
}

}  // namespace Carbon
