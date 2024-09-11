// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/driver.h"

#include <algorithm>
#include <memory>
#include <optional>

#include "common/command_line.h"
#include "common/version.h"
#include "common/vlog.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/Triple.h"
#include "toolchain/base/value_store.h"
#include "toolchain/check/check.h"
#include "toolchain/codegen/codegen.h"
#include "toolchain/diagnostics/sorting_diagnostic_consumer.h"
#include "toolchain/driver/clang_runner.h"
#include "toolchain/lex/lex.h"
#include "toolchain/lower/lower.h"
#include "toolchain/parse/parse.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/formatter.h"
#include "toolchain/sem_ir/inst_namer.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon {

struct Driver::Options {
  static const CommandLine::CommandInfo Info;

  enum class Subcommand : int8_t {
    Compile,
    Link,
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
                      compile_options.Build(sub_b, codegen_options);
                      sub_b.Do([&] { subcommand = Subcommand::Compile; });
                    });

    b.AddSubcommand(LinkOptions::Info, [&](CommandLine::CommandBuilder& sub_b) {
      link_options.Build(sub_b, codegen_options);
      sub_b.Do([&] { subcommand = Subcommand::Link; });
    });

    b.RequiresSubcommand();
  }

  bool verbose;
  Subcommand subcommand;

  CodegenOptions codegen_options;
  CompileOptions compile_options;
  LinkOptions link_options;
};

// Note that this is not constexpr so that it can include information generated
// in separate translation units and potentially overridden at link time in the
// version string.
const CommandLine::CommandInfo Driver::Options::Info = {
    .name = "carbon",
    .version = Version::ToolchainInfo,
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

auto Driver::ParseArgs(llvm::ArrayRef<llvm::StringRef> args, Options& options)
    -> CommandLine::ParseResult {
  return CommandLine::Parse(
      args, driver_env_.output_stream, driver_env_.error_stream, Options::Info,
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
    driver_env_.vlog_stream = &driver_env_.error_stream;
  }

  switch (options.subcommand) {
    case Options::Subcommand::Compile:
      return Compile(options.compile_options, options.codegen_options);
    case Options::Subcommand::Link:
      return Link(options.link_options, options.codegen_options);
  }
  llvm_unreachable("All subcommands handled!");
}

auto Driver::ValidateCompileOptions(const CompileOptions& options) const
    -> bool {
  using Phase = CompileOptions::Phase;
  switch (options.phase) {
    case Phase::Lex:
      if (options.dump_parse_tree) {
        driver_env_.error_stream
            << "ERROR: Requested dumping the parse tree but compile "
               "phase is limited to '"
            << options.phase << "'.\n";
        return false;
      }
      [[fallthrough]];
    case Phase::Parse:
      if (options.dump_sem_ir) {
        driver_env_.error_stream
            << "ERROR: Requested dumping the SemIR but compile phase "
               "is limited to '"
            << options.phase << "'.\n";
        return false;
      }
      [[fallthrough]];
    case Phase::Check:
      if (options.dump_llvm_ir) {
        driver_env_.error_stream
            << "ERROR: Requested dumping the LLVM IR but compile "
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
  explicit CompilationUnit(DriverEnv& driver_env, const CompileOptions& options,
                           const CodegenOptions& codegen_options,
                           DiagnosticConsumer* consumer,
                           llvm::StringRef input_filename)
      : driver_env_(&driver_env),
        options_(options),
        codegen_options_(codegen_options),
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
    LogCall("SourceBuffer::MakeFromFile", [&] {
      if (input_filename_ == "-") {
        source_ = SourceBuffer::MakeFromStdin(*consumer_);
      } else {
        source_ = SourceBuffer::MakeFromFile(driver_env_->fs, input_filename_,
                                             *consumer_);
      }
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

  auto input_filename() -> llvm::StringRef { return input_filename_; }
  auto success() -> bool { return success_; }
  auto has_source() -> bool { return source_.has_value(); }

 private:
  // Do codegen. Returns true on success.
  auto RunCodeGenHelper() -> bool {
    std::optional<CodeGen> codegen = CodeGen::Make(
        *module_, codegen_options_.target, driver_env_->error_stream);
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
      CARBON_VLOG("Writing output to: {0}\n", output_filename);

      std::error_code ec;
      llvm::raw_fd_ostream output_file(output_filename, ec,
                                       llvm::sys::fs::OF_None);
      if (ec) {
        driver_env_->error_stream << "ERROR: Could not open output file '"
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
  const CodegenOptions& codegen_options_;
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

auto Driver::Compile(const CompileOptions& options,
                     const CodegenOptions& codegen_options) -> RunResult {
  if (!ValidateCompileOptions(options)) {
    return {.success = false};
  }

  // Find the files comprising the prelude if we are importing it.
  // TODO: Replace this with a search for library api files in a
  // package-specific search path based on the library name.
  llvm::SmallVector<std::string> prelude;
  if (options.prelude_import && options.phase >= CompileOptions::Phase::Check) {
    if (auto find = driver_env_.installation->ReadPreludeManifest();
        find.ok()) {
      prelude = std::move(*find);
    } else {
      driver_env_.error_stream << "ERROR: " << find.error() << "\n";
      return {.success = false};
    }
  }

  // Prepare CompilationUnits before building scope exit handlers.
  StreamDiagnosticConsumer stream_consumer(driver_env_.error_stream);
  llvm::SmallVector<std::unique_ptr<CompilationUnit>> units;
  units.reserve(prelude.size() + options.input_filenames.size());

  // Add the prelude files.
  for (const auto& input_filename : prelude) {
    units.push_back(
        std::make_unique<CompilationUnit>(driver_env_, options, codegen_options,
                                          &stream_consumer, input_filename));
  }

  // Add the input source files.
  for (const auto& input_filename : options.input_filenames) {
    units.push_back(
        std::make_unique<CompilationUnit>(driver_env_, options, codegen_options,
                                          &stream_consumer, input_filename));
  }

  auto on_exit = llvm::make_scope_exit([&]() {
    // Finish compilation units. This flushes their diagnostics in the order in
    // which they were specified on the command line.
    for (auto& unit : units) {
      unit->PostCompile();
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
  CARBON_VLOG_TO(driver_env_.vlog_stream, "*** Check::CheckParseTrees ***\n");
  Check::CheckParseTrees(check_units, node_converters, options.prelude_import,
                         driver_env_.vlog_stream);
  CARBON_VLOG_TO(driver_env_.vlog_stream,
                 "*** Check::CheckParseTrees done ***\n");
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
    CARBON_VLOG_TO(driver_env_.vlog_stream,
                   "*** Stopping before lowering due to errors ***");
    return make_result();
  }

  // Lower.
  for (const auto& unit : units) {
    Check::SemIRDiagnosticConverter converter(node_converters,
                                              &**unit->GetCheckUnit().sem_ir);
    unit->RunLower(converter);
  }
  if (options.phase == CompileOptions::Phase::Lower) {
    return make_result();
  }
  CARBON_CHECK(options.phase == CompileOptions::Phase::CodeGen,
               "CodeGen should be the last stage");

  // Codegen.
  for (auto& unit : units) {
    unit->RunCodeGen();
  }
  return make_result();
}

static void AddOSFlags(llvm::StringRef target,
                       llvm::SmallVectorImpl<llvm::StringRef>& args) {
  llvm::Triple triple(target);
  switch (triple.getOS()) {
    case llvm::Triple::Darwin:
    case llvm::Triple::MacOSX:
      // On macOS we need to set the sysroot to a viable SDK. Currently, this
      // hard codes the path to be the unversioned symlink. The prefix is also
      // hard coded in Homebrew and so this seems likely to work reasonably
      // well. Homebrew and I suspect the Xcode Clang both have this hard coded
      // at build time, so this seems reasonably safe but we can revisit if/when
      // needed.
      args.push_back(
          "--sysroot=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk");
      // We also need to insist on a modern linker, otherwise the driver tries
      // too old and deprecated flags. The specific number here comes from an
      // inspection of the Clang driver source code to understand where features
      // were enabled, and this appears to be the latest version to control
      // driver behavior.
      //
      // TODO: We should replace this with use of `lld` eventually.
      args.push_back("-mlinker-version=705");
      break;

    default:
      // By default, just let the Clang driver handle everything.
      break;
  }
}

auto Driver::Link(const LinkOptions& options,
                  const CodegenOptions& codegen_options) -> RunResult {
  // TODO: Currently we use the Clang driver to link. This works well on Unix
  // OSes but we likely need to directly build logic to invoke `link.exe` on
  // Windows where `cl.exe` doesn't typically cover that logic.

  // Use a reasonably large small vector here to minimize allocations. We expect
  // to link reasonably large numbers of object files.
  llvm::SmallVector<llvm::StringRef, 128> clang_args;

  // We link using a C++ mode of the driver.
  clang_args.push_back("--driver-mode=g++");

  // Use LLD, which we provide in our install directory, for linking.
  clang_args.push_back("-fuse-ld=lld");

  // Disable linking the C++ standard library until can build and ship it as
  // part of the Carbon toolchain. This clearly won't work once we get into
  // interop, but for now it avoids spurious failures and distraction. The plan
  // is to build and bundle libc++ at which point we can replace this with
  // pointing at our bundled library.
  // TODO: Replace this when ready.
  clang_args.push_back("-nostdlib++");

  // Add OS-specific flags based on the target.
  AddOSFlags(codegen_options.target, clang_args);

  clang_args.push_back("-o");
  clang_args.push_back(options.output_filename);
  clang_args.append(options.object_filenames.begin(),
                    options.object_filenames.end());

  ClangRunner runner(driver_env_.installation, codegen_options.target,
                     driver_env_.vlog_stream);
  return {.success = runner.Run(clang_args)};
}

}  // namespace Carbon
