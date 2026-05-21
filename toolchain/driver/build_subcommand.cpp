// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/build_subcommand.h"

#include <filesystem>

#include "common/command_line.h"
#include "common/filesystem.h"
#include "common/hashing.h"
#include "common/pretty_stack_trace_function.h"
#include "common/vlog.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Target/TargetMachine.h"
#include "toolchain/check/check.h"
#include "toolchain/codegen/codegen.h"
#include "toolchain/diagnostics/sorting_consumer.h"
#include "toolchain/driver/clang_runner.h"
#include "toolchain/driver/driver_subcommand.h"
#include "toolchain/lex/lex.h"
#include "toolchain/lower/lower.h"
#include "toolchain/parse/parse.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon {

auto BuildSubcommandOptions::Build(CommandLine::CommandBuilder& b) -> void {
  compile.Build(b);
  b.AddStringOption(
      {
          .name = "output",
          .short_name = "o",
          .value_name = "FILE",
          .help = R"""(
The file name for the output binary. If none is specified `build` will use the
name of the first provided input file.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&output_filename); });
  b.AddStringPositionalArg(
      {
          .name = "EXTRA_CLANG_LINK_ARGS",
          .help = R"""(
Extra arguments to pass to Clang when forming the link command. This is
primarily useful for expanding `LDFLAGS` or other baseline linking flags in a
build system.

These can also be used to pass object files to the link in the event your build
system mixes object files and linker flags.
)""",
      },
      [&](auto& arg_b) { arg_b.Append(&extra_clang_link_args); });
  b.AddFlag(
      {
          .name = "use-temp-dir",
          .help = R"""(
Use a temporary directory for intermediate compilation artifacts.

When enabled (the default), carbon will compile all input files and necessary
dependencies into a temporary directory, before linking them into the final
output binary. If false, carbon will store the compilation artifacts as hashes
of the compiled input name in the current working directory.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Default(true);
        arg_b.Set(&use_temp_dir);
      });
}

static constexpr CommandLine::CommandInfo SubcommandInfo = {
    .name = "build",
    .help = R"""(
Compile and then link Carbon and C++ source code into a single executable.
)""",
};

BuildSubcommand::BuildSubcommand() : DriverSubcommand(SubcommandInfo) {}

auto BuildSubcommand::BuildOptions(CommandLine::CommandBuilder& b) -> void {
  options_.Build(b);
}

// The classes defined here are forks of the objects of the same name in
// the `compile` subcommand. The `build` subcommand offers fewer options
// for compilation, and expects to link these objects at the end. So the logic
// is both simpler and different, particularly around output file paths.
namespace {

class MultiUnitCache;

class CompilationUnit {
 public:
  // `driver_env`, `options`, `consumer`, and `target` must be non-null. The
  // file at the path to `input_filename` must be a regular file.
  explicit CompilationUnit(SemIR::CheckIRId check_ir_id, int total_ir_count,
                           DriverEnv* driver_env,
                           const BuildSubcommandOptions* options,
                           Diagnostics::Consumer* consumer,
                           llvm::StringRef input_filename,
                           const std::filesystem::path& output_directory,
                           const llvm::Target* target);

  // Sets the multi-unit cache and initializes dependent member state.
  auto SetMultiUnitCache(MultiUnitCache* cache) -> void;

  // Loads source and lexes it. Returns true on success.
  auto RunLex() -> void;

  // Parses tokens. Returns true on success.
  auto RunParse() -> void;

  // Returns information needed to check this unit.
  auto GetCheckUnit() -> Check::Unit;

  // Runs post-check logic. Returns true if checking succeeded for the IR.
  auto PostCheck() -> void;

  // Lower SemIR to LLVM IR.
  auto RunLower() -> void;

  // Runs the optimization pipeline.
  auto RunOptimize(const clang::CompilerInvocation& clang_invocation) -> void;

  auto RunCodeGen() -> void;

  // Runs post-compile logic. This is always called, and called after all other
  // actions on the CompilationUnit.
  auto PostCompile() -> void;

  // Flushes diagnostics, specifically as part of generating stack trace
  // information.
  auto FlushForStackTrace() -> void { consumer_->Flush(); }

  auto input_filename() -> llvm::StringRef { return input_filename_; }
  auto output_filename() -> llvm::StringRef { return output_filename_; }
  auto success() -> bool { return success_; }
  auto has_source() -> bool { return source_.has_value(); }
  auto get_trees_and_subtrees() -> Parse::GetTreeAndSubtreesFn {
    return *tree_and_subtrees_getter_;
  }

 private:
  // Do codegen. Returns true on success.
  auto RunCodeGenHelper() -> bool;

  // The TreeAndSubtrees is mainly used for debugging and diagnostics, and has
  // significant overhead. Avoid constructing it when unused.
  auto GetParseTreeAndSubtrees() -> const Parse::TreeAndSubtrees&;

  // Wraps a call with log statements to indicate start and end. Typically logs
  // with the actual function name, but marks timings with the appropriate
  // phase.
  auto LogCall(llvm::StringLiteral logging_label,
               llvm::StringLiteral timing_label,
               llvm::function_ref<auto()->void> fn) -> void;

  // Builds the LLVM target machine.
  auto MakeTargetMachine(const clang::CompilerInvocation& clang_invocation)
      -> void;

  // The index of the unit amongst all units.
  SemIR::CheckIRId check_ir_id_;
  // The number of units in total.
  int total_ir_count_;

  DriverEnv* driver_env_;
  const BuildSubcommandOptions* options_;
  const llvm::Target* target_;

  SharedValueStores value_stores_;

  // The input filename from the command line. For most diagnostics, we
  // typically use `source_->filename()`, which includes a `-` -> `<stdin>`
  // translation. However, logging and some diagnostics use the command line
  // argument.
  std::string input_filename_;
  // The temporary directory where we will store the compiled output of the
  // CompilationUnit
  std::filesystem::path output_directory_;
  // The output filename, computed as a hash of the input filename with a `.o`
  // extension.
  std::string output_filename_;

  // Copied from driver_ for CARBON_VLOG.
  llvm::raw_pwrite_stream* vlog_stream_;

  // Diagnostics are sent to consumer_, with optional sorting.
  std::optional<Diagnostics::SortingConsumer> sorting_consumer_;
  Diagnostics::Consumer* consumer_;

  bool success_ = true;

  // Initialized by `SetMultiUnitCache`.
  MultiUnitCache* cache_ = nullptr;

  // These are initialized as steps are run.
  std::optional<SourceBuffer> source_;
  std::optional<Lex::TokenizedBuffer> tokens_;
  std::optional<Parse::Tree> parse_tree_;
  std::optional<Parse::TreeAndSubtrees> parse_tree_and_subtrees_;
  std::optional<std::function<auto()->const Parse::TreeAndSubtrees&>>
      tree_and_subtrees_getter_;
  std::unique_ptr<llvm::LLVMContext> llvm_context_;
  std::optional<SemIR::File> sem_ir_;
  std::unique_ptr<llvm::Module> module_;
  std::unique_ptr<llvm::TargetMachine> target_machine_;
};

// Caches lists that are shared cross-unit. Accessors do lazy caching because
// they may not be used.
class MultiUnitCache {
 public:
  using TreeAndSubtreesGettersStore = Parse::GetTreeAndSubtreesStore;

  // This relies on construction after `units` are all initialized, which is
  // reflected by the `ArrayRef` here.
  explicit MultiUnitCache(
      llvm::ArrayRef<std::unique_ptr<CompilationUnit>> units)
      : units_(units) {}

  auto tree_and_subtrees_getters() -> const TreeAndSubtreesGettersStore& {
    if (!tree_and_subtrees_getters_) {
      tree_and_subtrees_getters_.emplace(
          TreeAndSubtreesGettersStore::MakeWithExplicitSize(units_.size(),
                                                            nullptr));
      for (const auto& [i, unit] : llvm::enumerate(units_)) {
        if (unit->has_source()) {
          tree_and_subtrees_getters_->Set(SemIR::CheckIRId(i),
                                          unit->get_trees_and_subtrees());
        }
      }
    }
    return *tree_and_subtrees_getters_;
  }

 private:
  // The units being compiled.
  llvm::ArrayRef<std::unique_ptr<CompilationUnit>> units_;

  // For each unit, the `TreeAndSubtrees` getter. Used by lowering.
  std::optional<TreeAndSubtreesGettersStore> tree_and_subtrees_getters_;
};

CompilationUnit::CompilationUnit(SemIR::CheckIRId check_ir_id,
                                 int total_ir_count, DriverEnv* driver_env,
                                 const BuildSubcommandOptions* options,
                                 Diagnostics::Consumer* consumer,
                                 llvm::StringRef input_filename,
                                 const std::filesystem::path& output_directory,
                                 const llvm::Target* target)
    : check_ir_id_(check_ir_id),
      total_ir_count_(total_ir_count),
      driver_env_(driver_env),
      options_(options),
      target_(target),
      input_filename_(input_filename),
      vlog_stream_(driver_env_->vlog_stream) {
  sorting_consumer_ = Diagnostics::SortingConsumer(*consumer);
  consumer_ = &*sorting_consumer_;

  // All input files are processed into a flat temporary output directory. In
  // order to avoid name collisions between input files with the same name but
  // different paths, we hash the entire path of the input file and use the hash
  // value as the name of the output file.
  output_filename_ =
      (output_directory /
       llvm::formatv("{0:x16}.o", HashValue(input_filename_)).str())
          .string();
}

auto CompilationUnit::SetMultiUnitCache(MultiUnitCache* cache) -> void {
  CARBON_CHECK(!cache_, "Called SetMultiUnitCache twice");
  cache_ = cache;
}

auto CompilationUnit::RunLex() -> void {
  CARBON_CHECK(cache_, "Must call SetMultiUnitCache first");
  CARBON_CHECK(!tokens_, "Called RunLex twice");

  LogCall("SourceBuffer::MakeFromFileOrStdin", "source", [&] {
    source_ = SourceBuffer::MakeFromFileOrStdin(*driver_env_->fs,
                                                input_filename_, *consumer_);
  });

  if (!source_) {
    success_ = false;
    return;
  }

  CARBON_VLOG("*** SourceBuffer ***\n```\n{0}\n```\n", source_->text());

  LogCall("Lex::Lex", "lex", [&] {
    Lex::LexOptions options;
    options.consumer = consumer_;
    options.vlog_stream = vlog_stream_;
    tokens_ = Lex::Lex(value_stores_, *source_, options);
  });
  if (tokens_->has_errors()) {
    success_ = false;
  }
}

auto CompilationUnit::RunParse() -> void {
  LogCall("Parse::Parse", "parse", [&] {
    Parse::ParseOptions options;
    options.consumer = consumer_;
    options.vlog_stream = vlog_stream_;
    parse_tree_ = Parse::Parse(*tokens_, options);
  });
  if (parse_tree_->has_errors()) {
    success_ = false;
  }
}

auto CompilationUnit::GetCheckUnit() -> Check::Unit {
  CARBON_CHECK(parse_tree_, "Must call RunParse first");
  CARBON_CHECK(!sem_ir_, "Called GetCheckUnit twice");

  tree_and_subtrees_getter_ = [this]() -> const Parse::TreeAndSubtrees& {
    return this->GetParseTreeAndSubtrees();
  };
  sem_ir_.emplace(&*parse_tree_, check_ir_id_, parse_tree_->packaging_decl(),
                  value_stores_, input_filename_);
  if (!llvm_context_) {
    llvm_context_ = std::make_unique<llvm::LLVMContext>();
  }
  return {.consumer = consumer_,
          .value_stores = &value_stores_,
          .timings = nullptr,
          .sem_ir = &*sem_ir_,
          .llvm_context = llvm_context_.get(),
          .total_ir_count = total_ir_count_};
}

auto CompilationUnit::PostCheck() -> void {
  CARBON_CHECK(sem_ir_, "Must call GetCheckUnit first");

  // We've finished all steps that can produce diagnostics. Emit the
  // diagnostics now, so that the developer sees them sooner and doesn't need
  // to wait for code generation.
  consumer_->Flush();

  if (sem_ir_->has_errors()) {
    success_ = false;
  }
}

auto CompilationUnit::RunLower() -> void {
  LogCall("Lower::LowerToLLVM", "lower", [&] {
    if (!llvm_context_) {
      llvm_context_ = std::make_unique<llvm::LLVMContext>();
    }
    Lower::LowerToLLVMOptions options;
    options.llvm_verifier_stream = driver_env_->error_stream;
    options.want_debug_info = options_->compile.include_debug_info;
    options.vlog_stream = vlog_stream_;
    options.opt_level = options_->compile.opt_level;
    module_ = Lower::LowerToLLVM(*llvm_context_, driver_env_->fs,
                                 cache_->tree_and_subtrees_getters(), *sem_ir_,
                                 total_ir_count_, options);
  });
}

auto CompilationUnit::MakeTargetMachine(
    const clang::CompilerInvocation& clang_invocation) -> void {
  CARBON_CHECK(module_, "Must call RunLower first");
  CARBON_CHECK(!target_machine_, "Should not call this multiple times");

  // Set the target on the module.
  // TODO: We should do this earlier. Lower should be passed the target triple
  // so it can create the module with this already set.
  llvm::Triple target_triple(options_->compile.codegen_options.target);
  module_->setTargetTriple(target_triple);

  // TODO: Provide flags to control these.
  constexpr llvm::StringLiteral CPU = "generic";
  constexpr llvm::StringLiteral Features = "";

  const auto& codegen_opts = clang_invocation.getCodeGenOpts();

  // TODO: Make the code in Clang's BackendUtil.cpp externally accessible and
  // call it from here. This is doing a subset of the same work to translate
  // Clang code generation options into target options.
  llvm::TargetOptions target_opts;
  target_opts.UseInitArray = codegen_opts.UseInitArray;
  target_opts.FunctionSections = codegen_opts.FunctionSections;
  target_opts.DataSections = codegen_opts.DataSections;
  target_opts.UniqueSectionNames = codegen_opts.UniqueSectionNames;
  target_machine_.reset(target_->createTargetMachine(
      target_triple, CPU, Features, target_opts, llvm::Reloc::PIC_));
}

auto CompilationUnit::RunOptimize(
    const clang::CompilerInvocation& clang_invocation) -> void {
  CARBON_CHECK(module_, "Must call RunLower first");

  // TODO: A lot of the work done here duplicates work done by Clang setting up
  // its pass manager. Moreover, we probably want to pick up Clang's
  // customizations and make use of its flags for controlling LLVM passes. We
  // should consider whether we would be better off running Clang's pass
  // pipeline rather than building one of our own, or factoring out enough of
  // Clang's pipeline builder that we can reuse and further customize it.

  MakeTargetMachine(clang_invocation);

  // TODO: There's no way to set these automatically from an
  // llvm::OptimizationLevel. Add such a mechanism to LLVM and use it from
  // here. For now we reconstruct what Clang does by default.
  llvm::PipelineTuningOptions pto;
  bool opt_for_speed =
      options_->compile.opt_level == Lower::OptimizationLevel::Speed;
  bool opt_for_size_or_speed =
      opt_for_speed ||
      options_->compile.opt_level == Lower::OptimizationLevel::Size;
  // Loop unrolling is enabled by `--optimize=size` but isn't actually performed
  // because we add `optsize` attributes to the function definitions we emit.
  pto.LoopUnrolling = opt_for_size_or_speed;
  pto.LoopInterleaving = opt_for_size_or_speed;
  pto.LoopVectorization = opt_for_speed;
  pto.SLPVectorization = opt_for_size_or_speed;

  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  llvm::PassInstrumentationCallbacks pic;

  // Register standard pass instrumentations. This adds support for things like
  // `-print-after-all`.
  llvm::StandardInstrumentations si(module_->getContext(),
                                    /*DebugLogging=*/false);
  si.registerCallbacks(pic);

  llvm::PassBuilder builder(target_machine_.get(), pto,
                            /*PGOOpt=*/std::nullopt, &pic);

  // TODO: Add an AssignmentTrackingPass for at least `--optimize=debug`.

  // Set up target library information and add an analysis pass to supply it.
  std::unique_ptr<llvm::TargetLibraryInfoImpl> tlii(llvm::driver::createTLII(
      module_->getTargetTriple(), llvm::driver::VectorLibrary::NoLibrary));
  fam.registerPass([&] { return llvm::TargetLibraryAnalysis(*tlii); });

  builder.registerModuleAnalyses(mam);
  builder.registerCGSCCAnalyses(cgam);
  builder.registerFunctionAnalyses(fam);
  builder.registerLoopAnalyses(lam);
  builder.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::ModulePassManager pass_manager = builder.buildPerModuleDefaultPipeline(
      SharedCompileOptions::GetLLVMOptimizationLevel(
          options_->compile.opt_level));

  LogCall("ModulePassManager::run", "optimize",
          [&] { pass_manager.run(*module_, mam); });
}

auto CompilationUnit::RunCodeGen() -> void {
  CARBON_CHECK(module_, "Must call RunLower first");
  LogCall("CodeGen", "codegen", [&] { success_ = RunCodeGenHelper(); });
}

auto CompilationUnit::PostCompile() -> void {
  // The diagnostics consumer must be flushed before compilation artifacts are
  // destructed, because diagnostics can refer to their state.
  consumer_->Flush();
}

auto CompilationUnit::RunCodeGenHelper() -> bool {
  CARBON_CHECK(module_, "Must call RunLower first");
  CARBON_CHECK(target_machine_, "Must call MakeTargetMachine first");

  CodeGen codegen(module_.get(), target_machine_.get(), consumer_);

  CARBON_VLOG("Writing output to: {0}\n", output_filename_);
  std::error_code ec;
  llvm::raw_fd_ostream output_file(output_filename_, ec,
                                   llvm::sys::fs::OF_None);
  if (ec) {
    // TODO: Consider rephrasing the diagnostic to use the file as the `Emit`
    CARBON_DIAGNOSTIC(BuildOutputFileOpenError, Error,
                      "could not open output file `{0}`: {1}", std::string,
                      std::string);
    driver_env_->emitter.Emit(BuildOutputFileOpenError, output_filename_,
                              ec.message());
    return false;
  }

  if (!codegen.EmitObject(output_file)) {
    return false;
  }
  return true;
}

auto CompilationUnit::GetParseTreeAndSubtrees()
    -> const Parse::TreeAndSubtrees& {
  if (!parse_tree_and_subtrees_) {
    parse_tree_and_subtrees_ = Parse::TreeAndSubtrees(*tokens_, *parse_tree_);
  }
  return *parse_tree_and_subtrees_;
}

auto CompilationUnit::LogCall(llvm::StringLiteral logging_label,
                              llvm::StringLiteral timing_label,
                              llvm::function_ref<auto()->void> fn) -> void {
  PrettyStackTraceFunction trace_file([&](llvm::raw_ostream& out) {
    out << "Filename: " << input_filename_ << "\n";
  });
  CARBON_VLOG("*** {0}: {1} ***\n", logging_label, input_filename_);
  Timings::ScopedTiming timing(nullptr, timing_label);
  fn();
  CARBON_VLOG("*** {0} done ***\n", logging_label);
}

}  // namespace

auto BuildSubcommand::Run(DriverEnv& driver_env) -> DriverResult {
  const llvm::Target* target;
  if (auto t = options_.compile.ValidateTarget(driver_env.emitter); t.ok()) {
    target = *t;
  } else {
    return {.success = false};
  }

  std::shared_ptr<clang::CompilerInvocation> clang_invocation;
  {
    if (driver_env.fuzzing && !options_.compile.clang_args.empty()) {
      // Parsing specific Clang arguments can reach deep into
      // external libraries that aren't fuzz clean.
      TestAndDiagnoseIfFuzzingExternalLibraries(driver_env, "build");
      return {.success = false};
    }

    if (auto i = options_.compile.BuildClangInvocation(driver_env); i.ok()) {
      clang_invocation = *i;
    } else {
      return {.success = false};
    }
  }

  // TODO: automatic prelude resolution.
  llvm::SmallVector<std::string> prelude;
  if (auto find = driver_env.installation->ReadPreludeManifest(); find.ok()) {
    prelude = std::move(*find);
  } else {
    // TODO: Change ReadPreludeManifest to produce diagnostics.
    CARBON_DIAGNOSTIC(BuildPreludeManifestError, Error, "{0}", std::string);
    driver_env.emitter.Emit(BuildPreludeManifestError,
                            PrintToString(find.error()));
    return {.success = false};
  }

  std::optional<Filesystem::RemovingDir> temp_dir = std::nullopt;
  if (options_.use_temp_dir) {
    if (auto d = Filesystem::MakeTmpDir(); d.ok()) {
      temp_dir = std::move(*d);
    } else {
      CARBON_DIAGNOSTIC(BuildTempDirectoryCreationError, Error, "{0}",
                        std::string);
      driver_env.emitter.Emit(BuildTempDirectoryCreationError,
                              PrintToString(d.error()));
      return {.success = false};
    }
  }

  // Prepare CompilationUnits before building scope exit handlers.
  llvm::SmallVector<std::unique_ptr<CompilationUnit>> units;
  int unit_index = -1;
  int total_unit_count =
      prelude.size() + options_.compile.input_filenames.size();
  auto unit_builder = [&](llvm::StringRef input_filename) {
    ++unit_index;
    return std::make_unique<CompilationUnit>(
        SemIR::CheckIRId(unit_index), total_unit_count, &driver_env, &options_,
        &driver_env.consumer, input_filename, temp_dir ? temp_dir->path() : "",
        target);
  };
  llvm::append_range(units, llvm::map_range(prelude, unit_builder));
  // Save the unit index of the first input filename, in case we need to compute
  // the output filename from this.
  auto input_filenames_index = units.size();
  llvm::append_range(
      units, llvm::map_range(options_.compile.input_filenames, unit_builder));
  CARBON_CHECK(units.size() == static_cast<size_t>(total_unit_count));

  // Add the cache to all units. This must be done after all units are
  // created.
  MultiUnitCache cache(units);
  for (auto& unit : units) {
    unit->SetMultiUnitCache(&cache);
  }

  auto on_exit = llvm::scope_exit([&]() {
    // Finish compilation units. This flushes their diagnostics in the order
    // in which they were specified on the command line.
    for (auto& unit : units) {
      unit->PostCompile();
    }

    // Clean up the temporary directory created for compile results.
    if (temp_dir) {
      auto remove_result = std::move(*temp_dir).Remove();
      if (!remove_result.ok()) {
        CARBON_DIAGNOSTIC(BuildTempDirectoryDeletionError, Error, "{0}",
                          std::string);
        driver_env.emitter.Emit(BuildTempDirectoryDeletionError,
                                PrintToString(remove_result.error()));
      }
    }

    driver_env.consumer.Flush();
  });

  // Returns a DriverResult object. Called whenever any of the compilation steps
  // in Build return.
  auto make_result = [&]() {
    DriverResult result = {.success = true};

    for (const auto& unit : units) {
      result.success &= unit->success();
      result.per_file_success.push_back(
          {unit->input_filename().str(), unit->success()});
    }
    return result;
  };

  auto has_unit_error = [&]() {
    return llvm::any_of(units,
                        [&](const auto& unit) { return !unit->success(); });
  };

  // Lex.
  for (auto& unit : units) {
    unit->RunLex();
  }
  // Parse and check phases examine `has_source` because they want to proceed
  // if lex failed, but not if source doesn't exist. Later steps are skipped
  // if anything failed, so don't need this.

  // Parse.
  for (auto& unit : units) {
    if (unit->has_source()) {
      unit->RunParse();
    }
  }

  // Gather Check::Units.
  llvm::SmallVector<Check::Unit> check_units;
  check_units.reserve(units.size());
  for (auto& unit : units) {
    if (unit->has_source()) {
      check_units.push_back(unit->GetCheckUnit());
    }
  }

  // Execute the actual checking.
  Check::CheckParseTreesOptions options;
  options.prelude_import = true;
  options.vlog_stream = driver_env.vlog_stream;
  options.fuzzing = driver_env.fuzzing;
  // We're not dumping anything, so don't include anything in dumps. Check
  // segfaults without this.
  auto include_in_dumps =
      FixedSizeValueStore<SemIR::CheckIRId, bool>::MakeWithExplicitSize(
          units.size(), false);
  options.include_in_dumps = &include_in_dumps;
  Check::CheckParseTrees(check_units, cache.tree_and_subtrees_getters(),
                         driver_env.fs, options, clang_invocation);
  for (auto& unit : units) {
    if (unit->has_source()) {
      unit->PostCheck();
    }
  }

  // Unlike previous steps, errors block further progress.
  if (has_unit_error()) {
    CARBON_VLOG_TO(driver_env.vlog_stream,
                   "build halted due to errors during check");
    return make_result();
  }

  // Lower and optimize.
  for (const auto& unit : units) {
    unit->RunLower();
    unit->RunOptimize(*clang_invocation);
  }

  if (has_unit_error()) {
    CARBON_VLOG_TO(driver_env.vlog_stream,
                   "build halted due to errors during lowering/optimization");
    return make_result();
  }

  for (const auto& unit : units) {
    unit->RunCodeGen();
  }

  if (has_unit_error()) {
    CARBON_VLOG_TO(driver_env.vlog_stream,
                   "build halted due to errors during code generation");
    return make_result();
  }

  // We've successfully compiled the inputs files, time to link them.
  llvm::SmallVector<llvm::StringRef> clang_link_args;

  // We link using a C++ mode of the driver.
  clang_link_args.push_back("--driver-mode=g++");

  // Pass the target down to Clang to pick up the correct defaults.
  std::string target_arg =
      llvm::formatv("--target={0}", options_.compile.codegen_options.target)
          .str();
  clang_link_args.push_back(target_arg);

  llvm::SmallString<256> output_filename;
  if (!options_.output_filename.empty()) {
    clang_link_args.push_back("-o");
    clang_link_args.push_back(options_.output_filename);
  } else {
    output_filename = llvm::sys::path::filename(
        units[input_filenames_index]->input_filename());
    llvm::sys::path::replace_extension(output_filename, "");
    clang_link_args.push_back("-o");
    clang_link_args.push_back(output_filename);
  }

  // Note that we append any extra Clang args before our object filenames. This
  // allows us to propagate object filenames that collide with Clang flags using
  // `--` before the filenames. While in theory, this could create a problem in
  // the presence of mixtures of object files in the two lists and the order
  // being dependent, we don't expect that in practice.
  clang_link_args.append(options_.extra_clang_link_args.begin(),
                         options_.extra_clang_link_args.end());
  clang_link_args.push_back("--");
  auto input_builder = [&](std::unique_ptr<CompilationUnit>& unit) {
    return unit->output_filename();
  };
  append_range(clang_link_args, llvm::map_range(units, input_builder));

  CARBON_VLOG_TO(driver_env.vlog_stream,
                 "*** Build Clang link call with these arguments:\n");
  for (auto a : clang_link_args) {
    CARBON_VLOG_TO(driver_env.vlog_stream, "    '{0}',\n", a);
  }

  ClangRunner runner(driver_env.installation, driver_env.fs,
                     driver_env.vlog_stream);
  // Don't run Clang when fuzzing, it is known to not be reliable under fuzzing
  // due to many unfixed issues.
  if (TestAndDiagnoseIfFuzzingExternalLibraries(driver_env, "clang")) {
    return {.success = false};
  }

  // Question: We're including some runtime stuff during compilation, is this
  // redundant?
  ErrorOr<bool> run_result =
      driver_env.prebuilt_runtimes
          ? runner.RunWithPrebuiltRuntimes(clang_link_args,
                                           *driver_env.prebuilt_runtimes,
                                           driver_env.enable_leaking)
      : driver_env.build_runtimes_on_demand
          ? runner.Run(clang_link_args, driver_env.runtimes_cache,
                       *driver_env.thread_pool, driver_env.enable_leaking)
          : runner.RunWithNoRuntimes(clang_link_args,
                                     driver_env.enable_leaking);

  if (!run_result.ok()) {
    // This is not a Clang failure, but a failure to even run Clang, so we need
    // to diagnose it here.
    CARBON_DIAGNOSTIC(BuildFailureRunningClangToLink, Error,
                      "failure running `clang` to perform linking: {0}",
                      std::string);
    driver_env.emitter.Emit(BuildFailureRunningClangToLink,
                            run_result.error().message());
  }

  // Successfully ran Clang to perform the link, return its result.
  return {.success = *run_result};
}

}  // namespace Carbon
