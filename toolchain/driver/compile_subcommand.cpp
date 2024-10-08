// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/compile_subcommand.h"

#include "common/vlog.h"
#include "llvm/ADT/ScopeExit.h"
#include "toolchain/base/pretty_stack_trace_function.h"
#include "toolchain/check/check.h"
#include "toolchain/codegen/codegen.h"
#include "toolchain/diagnostics/sorting_diagnostic_consumer.h"
#include "toolchain/lex/lex.h"
#include "toolchain/lower/lower.h"
#include "toolchain/parse/parse.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/formatter.h"
#include "toolchain/sem_ir/inst_namer.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon {

auto operator<<(llvm::raw_ostream& out, CompileOptions::Phase phase)
    -> llvm::raw_ostream& {
  switch (phase) {
    case CompileOptions::Phase::Lex:
      out << "lex";
      break;
    case CompileOptions::Phase::Parse:
      out << "parse";
      break;
    case CompileOptions::Phase::Check:
      out << "check";
      break;
    case CompileOptions::Phase::Lower:
      out << "lower";
      break;
    case CompileOptions::Phase::CodeGen:
      out << "codegen";
      break;
  }
  return out;
}

constexpr CommandLine::CommandInfo CompileOptions::Info = {
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

auto CompileOptions::Build(CommandLine::CommandBuilder& b) -> void {
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

  // Include the common code generation options at this point to render it
  // after the more common options above, but before the more unusual options
  // below.
  codegen_options.Build(b);

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
          .name = "dump-mem-usage",
          .help = R"""(
Dumps the amount of memory used.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&dump_mem_usage); });
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
  b.AddFlag(
      {
          .name = "debug-info",
          .help = R"""(
Emit DWARF debug information.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Default(true);
        arg_b.Set(&include_debug_info);
      });
}

auto CompileSubcommand::ValidateOptions(DriverEnv& driver_env) const -> bool {
  using Phase = CompileOptions::Phase;
  switch (options_.phase) {
    case Phase::Lex:
      if (options_.dump_parse_tree) {
        driver_env.error_stream
            << "error: requested dumping the parse tree but compile "
               "phase is limited to '"
            << options_.phase << "'\n";
        return false;
      }
      [[fallthrough]];
    case Phase::Parse:
      if (options_.dump_sem_ir) {
        driver_env.error_stream
            << "error: requested dumping the SemIR but compile phase "
               "is limited to '"
            << options_.phase << "'\n";
        return false;
      }
      [[fallthrough]];
    case Phase::Check:
      if (options_.dump_llvm_ir) {
        driver_env.error_stream
            << "error: requested dumping the LLVM IR but compile "
               "phase is limited to '"
            << options_.phase << "'\n";
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

namespace {
// Ties together information for a file being compiled.
// TODO: Refactor this because it's a long class to have function definitions
// inline.
class CompilationUnit {
 public:
  explicit CompilationUnit(DriverEnv& driver_env, const CompileOptions& options,
                           DiagnosticConsumer* consumer,
                           llvm::StringRef input_filename)
      : driver_env_(&driver_env),
        options_(options),
        input_filename_(input_filename),
        vlog_stream_(driver_env_->vlog_stream) {
    if (vlog_stream_ != nullptr || options_.stream_errors) {
      consumer_ = consumer;
    } else {
      sorting_consumer_ = SortingDiagnosticConsumer(*consumer);
      consumer_ = &*sorting_consumer_;
    }
    if (options_.dump_mem_usage && IncludeInDumps()) {
      mem_usage_ = MemUsage();
    }
  }

  // Loads source and lexes it. Returns true on success.
  auto RunLex() -> void {
    LogCall("SourceBuffer::MakeFromFileOrStdin", [&] {
      source_ = SourceBuffer::MakeFromFileOrStdin(driver_env_->fs,
                                                  input_filename_, *consumer_);
    });
    if (mem_usage_) {
      mem_usage_->Add("source_", source_->text().size(),
                      source_->text().size());
    }
    if (!source_) {
      success_ = false;
      return;
    }
    CARBON_VLOG("*** SourceBuffer ***\n```\n{0}\n```\n", source_->text());

    LogCall("Lex::Lex",
            [&] { tokens_ = Lex::Lex(value_stores_, *source_, *consumer_); });
    if (options_.dump_tokens && IncludeInDumps()) {
      consumer_->Flush();
      driver_env_->output_stream << tokens_;
    }
    if (mem_usage_) {
      mem_usage_->Collect("tokens_", *tokens_);
    }
    CARBON_VLOG("*** Lex::TokenizedBuffer ***\n{0}", tokens_);
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
      const auto& tree_and_subtrees = GetParseTreeAndSubtrees();
      if (options_.preorder_parse_tree) {
        tree_and_subtrees.PrintPreorder(driver_env_->output_stream);
      } else {
        tree_and_subtrees.Print(driver_env_->output_stream);
      }
    }
    if (mem_usage_) {
      mem_usage_->Collect("parse_tree_", *parse_tree_);
    }
    CARBON_VLOG("*** Parse::Tree ***\n{0}", parse_tree_);
    if (parse_tree_->has_errors()) {
      success_ = false;
    }
  }

  // Returns information needed to check this unit.
  auto GetCheckUnit() -> Check::Unit {
    CARBON_CHECK(parse_tree_);
    return {
        .value_stores = &value_stores_,
        .tokens = &*tokens_,
        .parse_tree = &*parse_tree_,
        .consumer = consumer_,
        .get_parse_tree_and_subtrees = [&]() -> const Parse::TreeAndSubtrees& {
          return GetParseTreeAndSubtrees();
        },
        .sem_ir = &sem_ir_};
  }

  // Runs post-check logic. Returns true if checking succeeded for the IR.
  auto PostCheck() -> void {
    CARBON_CHECK(sem_ir_);

    // We've finished all steps that can produce diagnostics. Emit the
    // diagnostics now, so that the developer sees them sooner and doesn't need
    // to wait for code generation.
    consumer_->Flush();

    if (mem_usage_) {
      mem_usage_->Collect("sem_ir_", *sem_ir_);
    }

    if (options_.dump_raw_sem_ir && IncludeInDumps()) {
      CARBON_VLOG("*** Raw SemIR::File ***\n{0}\n", *sem_ir_);
      sem_ir_->Print(driver_env_->output_stream, options_.builtin_sem_ir);
      if (options_.dump_sem_ir) {
        driver_env_->output_stream << "\n";
      }
    }

    bool print = options_.dump_sem_ir && IncludeInDumps();
    if (vlog_stream_ || print) {
      SemIR::Formatter formatter(*tokens_, *parse_tree_, *sem_ir_);
      if (vlog_stream_) {
        CARBON_VLOG("*** SemIR::File ***\n");
        formatter.Print(*vlog_stream_);
      }
      if (print) {
        formatter.Print(driver_env_->output_stream);
      }
    }
    if (sem_ir_->has_errors()) {
      success_ = false;
    }
  }

  // Lower SemIR to LLVM IR.
  auto RunLower(const Check::SemIRDiagnosticConverter& converter) -> void {
    CARBON_CHECK(sem_ir_);

    LogCall("Lower::LowerToLLVM", [&] {
      llvm_context_ = std::make_unique<llvm::LLVMContext>();
      // TODO: Consider disabling instruction naming by default if we're not
      // producing textual LLVM IR.
      SemIR::InstNamer inst_namer(*tokens_, *parse_tree_, *sem_ir_);
      module_ = Lower::LowerToLLVM(*llvm_context_, options_.include_debug_info,
                                   converter, input_filename_, *sem_ir_,
                                   &inst_namer, vlog_stream_);
    });
    if (vlog_stream_) {
      CARBON_VLOG("*** llvm::Module ***\n");
      module_->print(*vlog_stream_, /*AAW=*/nullptr,
                     /*ShouldPreserveUseListOrder=*/false,
                     /*IsForDebug=*/true);
    }
    if (options_.dump_llvm_ir && IncludeInDumps()) {
      module_->print(driver_env_->output_stream, /*AAW=*/nullptr,
                     /*ShouldPreserveUseListOrder=*/true);
    }
  }

  auto RunCodeGen() -> void {
    CARBON_CHECK(module_);
    LogCall("CodeGen", [&] { success_ = RunCodeGenHelper(); });
  }

  // Runs post-compile logic. This is always called, and called after all other
  // actions on the CompilationUnit.
  auto PostCompile() -> void {
    if (options_.dump_shared_values && IncludeInDumps()) {
      Yaml::Print(driver_env_->output_stream,
                  value_stores_.OutputYaml(input_filename_));
    }
    if (mem_usage_) {
      mem_usage_->Collect("value_stores_", value_stores_);
      Yaml::Print(driver_env_->output_stream,
                  mem_usage_->OutputYaml(input_filename_));
    }

    // The diagnostics consumer must be flushed before compilation artifacts are
    // destructed, because diagnostics can refer to their state.
    consumer_->Flush();
  }

  // Flushes diagnostics, specifically as part of generating stack trace
  // information.
  auto FlushForStackTrace() -> void { consumer_->Flush(); }

  auto input_filename() -> llvm::StringRef { return input_filename_; }
  auto success() -> bool { return success_; }
  auto has_source() -> bool { return source_.has_value(); }

 private:
  // Do codegen. Returns true on success.
  auto RunCodeGenHelper() -> bool {
    std::optional<CodeGen> codegen = CodeGen::Make(
        *module_, options_.codegen_options.target, driver_env_->error_stream);
    if (!codegen) {
      return false;
    }
    if (vlog_stream_) {
      CARBON_VLOG("*** Assembly ***\n");
      codegen->EmitAssembly(*vlog_stream_);
    }

    if (options_.output_filename == "-") {
      // TODO: the output file name, forcing object output, and requesting
      // textual assembly output are all somewhat linked flags. We should add
      // some validation that they are used correctly.
      if (options_.force_obj_output) {
        if (!codegen->EmitObject(driver_env_->output_stream)) {
          return false;
        }
      } else {
        if (!codegen->EmitAssembly(driver_env_->output_stream)) {
          return false;
        }
      }
    } else {
      llvm::SmallString<256> output_filename = options_.output_filename;
      if (output_filename.empty()) {
        if (!source_->is_regular_file()) {
          // Don't invent file names like `-.o` or `/dev/stdin.o`.
          driver_env_->error_stream
              << "error: output file name must be specified for input '"
              << input_filename_ << "' that is not a regular file\n";
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
      CARBON_VLOG("Writing output to: {0}\n", output_filename);

      std::error_code ec;
      llvm::raw_fd_ostream output_file(output_filename, ec,
                                       llvm::sys::fs::OF_None);
      if (ec) {
        driver_env_->error_stream << "error: could not open output file '"
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

  // The TreeAndSubtrees is mainly used for debugging and diagnostics, and has
  // significant overhead. Avoid constructing it when unused.
  auto GetParseTreeAndSubtrees() -> const Parse::TreeAndSubtrees& {
    if (!parse_tree_and_subtrees_) {
      parse_tree_and_subtrees_ = Parse::TreeAndSubtrees(*tokens_, *parse_tree_);
      if (mem_usage_) {
        mem_usage_->Collect("parse_tree_and_subtrees_",
                            *parse_tree_and_subtrees_);
      }
    }
    return *parse_tree_and_subtrees_;
  }

  // Wraps a call with log statements to indicate start and end.
  auto LogCall(llvm::StringLiteral label, llvm::function_ref<void()> fn)
      -> void {
    CARBON_VLOG("*** {0}: {1} ***\n", label, input_filename_);
    fn();
    CARBON_VLOG("*** {0} done ***\n", label);
  }

  // Returns true if the file can be dumped.
  auto IncludeInDumps() const -> bool {
    return options_.exclude_dump_file_prefix.empty() ||
           !input_filename_.starts_with(options_.exclude_dump_file_prefix);
  }

  DriverEnv* driver_env_;
  SharedValueStores value_stores_;
  const CompileOptions& options_;
  std::string input_filename_;

  // Copied from driver_ for CARBON_VLOG.
  llvm::raw_pwrite_stream* vlog_stream_;

  // Diagnostics are sent to consumer_, with optional sorting.
  std::optional<SortingDiagnosticConsumer> sorting_consumer_;
  DiagnosticConsumer* consumer_;

  bool success_ = true;

  // Tracks memory usage of the compile.
  std::optional<MemUsage> mem_usage_;

  // These are initialized as steps are run.
  std::optional<SourceBuffer> source_;
  std::optional<Lex::TokenizedBuffer> tokens_;
  std::optional<Parse::Tree> parse_tree_;
  std::optional<Parse::TreeAndSubtrees> parse_tree_and_subtrees_;
  std::optional<SemIR::File> sem_ir_;
  std::unique_ptr<llvm::LLVMContext> llvm_context_;
  std::unique_ptr<llvm::Module> module_;
};
}  // namespace

auto CompileSubcommand::Run(DriverEnv& driver_env) -> DriverResult {
  if (!ValidateOptions(driver_env)) {
    return {.success = false};
  }

  // Find the files comprising the prelude if we are importing it.
  // TODO: Replace this with a search for library api files in a
  // package-specific search path based on the library name.
  llvm::SmallVector<std::string> prelude;
  if (options_.prelude_import &&
      options_.phase >= CompileOptions::Phase::Check) {
    if (auto find = driver_env.installation->ReadPreludeManifest(); find.ok()) {
      prelude = std::move(*find);
    } else {
      driver_env.error_stream << "error: " << find.error() << "\n";
      return {.success = false};
    }
  }

  // Prepare CompilationUnits before building scope exit handlers.
  StreamDiagnosticConsumer stream_consumer(driver_env.error_stream);
  llvm::SmallVector<std::unique_ptr<CompilationUnit>> units;
  units.reserve(prelude.size() + options_.input_filenames.size());

  // Add the prelude files.
  for (const auto& input_filename : prelude) {
    units.push_back(std::make_unique<CompilationUnit>(
        driver_env, options_, &stream_consumer, input_filename));
  }

  // Add the input source files.
  for (const auto& input_filename : options_.input_filenames) {
    units.push_back(std::make_unique<CompilationUnit>(
        driver_env, options_, &stream_consumer, input_filename));
  }

  auto on_exit = llvm::make_scope_exit([&]() {
    // Finish compilation units. This flushes their diagnostics in the order in
    // which they were specified on the command line.
    for (auto& unit : units) {
      unit->PostCompile();
    }

    stream_consumer.Flush();
  });

  PrettyStackTraceFunction flush_on_crash([&](llvm::raw_ostream& out) {
    // When crashing, flush diagnostics. If sorting diagnostics, they can be
    // redirected to the crash stream; if streaming, the original stream is
    // flushed.
    // TODO: Eventually we'll want to limit the count.
    if (options_.stream_errors) {
      out << "Flushing diagnostics\n";
    } else {
      out << "Pending diagnostics:\n";
      stream_consumer.set_stream(&out);
    }

    for (auto& unit : units) {
      unit->FlushForStackTrace();
    }
    stream_consumer.Flush();
    stream_consumer.set_stream(&driver_env.error_stream);
  });

  // Returns a DriverResult object. Called whenever Compile returns.
  auto make_result = [&]() {
    DriverResult result = {.success = true};
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
  if (options_.phase == CompileOptions::Phase::Lex) {
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
  if (options_.phase == CompileOptions::Phase::Parse) {
    return make_result();
  }

  // Check.
  SharedValueStores builtin_value_stores;
  llvm::SmallVector<Check::Unit> check_units;
  for (auto& unit : units) {
    if (unit->has_source()) {
      check_units.push_back(unit->GetCheckUnit());
    }
  }
  llvm::SmallVector<Parse::NodeLocConverter> node_converters;
  node_converters.reserve(check_units.size());
  for (auto& unit : check_units) {
    node_converters.emplace_back(unit.tokens, unit.tokens->source().filename(),
                                 unit.get_parse_tree_and_subtrees);
  }
  CARBON_VLOG_TO(driver_env.vlog_stream, "*** Check::CheckParseTrees ***\n");
  Check::CheckParseTrees(check_units, node_converters, options_.prelude_import,
                         driver_env.vlog_stream);
  CARBON_VLOG_TO(driver_env.vlog_stream,
                 "*** Check::CheckParseTrees done ***\n");
  for (auto& unit : units) {
    if (unit->has_source()) {
      unit->PostCheck();
    }
  }
  if (options_.phase == CompileOptions::Phase::Check) {
    return make_result();
  }

  // Unlike previous steps, errors block further progress.
  if (std::any_of(units.begin(), units.end(),
                  [&](const auto& unit) { return !unit->success(); })) {
    CARBON_VLOG_TO(driver_env.vlog_stream,
                   "*** Stopping before lowering due to errors ***");
    return make_result();
  }

  // Lower.
  for (const auto& unit : units) {
    Check::SemIRDiagnosticConverter converter(node_converters,
                                              &**unit->GetCheckUnit().sem_ir);
    unit->RunLower(converter);
  }
  if (options_.phase == CompileOptions::Phase::Lower) {
    return make_result();
  }
  CARBON_CHECK(options_.phase == CompileOptions::Phase::CodeGen,
               "CodeGen should be the last stage");

  // Codegen.
  for (auto& unit : units) {
    unit->RunCodeGen();
  }
  return make_result();
}

}  // namespace Carbon
