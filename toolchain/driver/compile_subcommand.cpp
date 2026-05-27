// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/compile_subcommand.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "common/pretty_stack_trace_function.h"
#include "common/vlog.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "toolchain/base/clang_invocation.h"
#include "toolchain/base/timings.h"
#include "toolchain/check/check.h"
#include "toolchain/codegen/codegen.h"
#include "toolchain/diagnostics/emitter.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/diagnostics/sorting_consumer.h"
#include "toolchain/lex/lex.h"
#include "toolchain/lower/lower.h"
#include "toolchain/parse/parse.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon {

static constexpr CommandLine::CommandInfo SubcommandInfo = {
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

CompileSubcommand::CompileSubcommand() : DriverSubcommand(SubcommandInfo) {}

auto CompileSubcommand::ValidateOptions(
    Diagnostics::NoLocEmitter& emitter) const -> bool {
  CARBON_DIAGNOSTIC(
      CompilePhaseFlagConflict, Error,
      "requested dumping {0} but compile phase is limited to `{1}`",
      std::string, std::string);
  using Phase = CompileOptions::Phase;
  switch (options_.phase) {
    case Phase::Lex:
      if (options_.dump_parse_tree) {
        emitter.Emit(CompilePhaseFlagConflict, "parse tree",
                     CompileOptions::PhaseToString(options_.phase));
        return false;
      }
      [[fallthrough]];
    case Phase::Parse:
      if (options_.dump_sem_ir) {
        emitter.Emit(CompilePhaseFlagConflict, "SemIR",
                     CompileOptions::PhaseToString(options_.phase));
        return false;
      }
      if (options_.dump_cpp_ast) {
        emitter.Emit(CompilePhaseFlagConflict, "C++ AST",
                     CompileOptions::PhaseToString(options_.phase));
        return false;
      }
      [[fallthrough]];
    case Phase::Check:
      if (options_.dump_llvm_ir) {
        emitter.Emit(CompilePhaseFlagConflict, "LLVM IR",
                     CompileOptions::PhaseToString(options_.phase));
        return false;
      }
      [[fallthrough]];
    case Phase::Lower:
    case Phase::Optimize:
    case Phase::CodeGen:
      // Everything can be dumped in these phases.
      break;
  }
  return true;
}

namespace {

class MultiUnitCache;

// Ties together information for a file being compiled.
class CompilationUnit {
 public:
  // `driver_env`, `options`, `consumer`, and `target` must be non-null.
  explicit CompilationUnit(SemIR::CheckIRId check_ir_id, int total_ir_count,
                           DriverEnv* driver_env, const CompileOptions* options,
                           Diagnostics::Consumer* consumer,
                           llvm::StringRef input_filename,
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

  // Runs post-lowering-to-LLVM-IR logic. This is always called if we do any
  // lowering work, after we've finished building the IR in RunLower() and,
  // optionally, RunOptimize().
  auto PostLower() -> void;

  auto RunCodeGen() -> void;

  // Runs post-compile logic. This is always called, and called after all other
  // actions on the CompilationUnit.
  auto PostCompile() -> void;

  // Flushes diagnostics, specifically as part of generating stack trace
  // information.
  auto FlushForStackTrace() -> void { consumer_->Flush(); }

  auto input_filename() -> llvm::StringRef { return input_filename_; }
  auto has_include_in_dumps() -> bool {
    return tokens_ && tokens_->has_include_in_dumps();
  }
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

  // Returns true if the current file should be included in debug dumps.
  auto IncludeInDumps() -> bool;

  // Builds the LLVM target machine.
  auto MakeTargetMachine(const clang::CompilerInvocation& clang_invocation)
      -> void;

  // The index of the unit amongst all units.
  SemIR::CheckIRId check_ir_id_;
  // The number of units in total.
  int total_ir_count_;

  DriverEnv* driver_env_;
  const CompileOptions* options_;
  const llvm::Target* target_;

  SharedValueStores value_stores_;

  // The input filename from the command line. For most diagnostics, we
  // typically use `source_->filename()`, which includes a `-` -> `<stdin>`
  // translation. However, logging and some diagnostics use the command line
  // argument.
  std::string input_filename_;

  // Copied from driver_ for CARBON_VLOG.
  llvm::raw_pwrite_stream* vlog_stream_;

  // Diagnostics are sent to consumer_, with optional sorting.
  std::optional<Diagnostics::SortingConsumer> sorting_consumer_;
  Diagnostics::Consumer* consumer_;

  bool success_ = true;

  // Initialized by `SetMultiUnitCache`.
  MultiUnitCache* cache_ = nullptr;
  // Tracks memory usage of the compile.
  std::optional<MemUsage> mem_usage_;
  // Tracks timings of the compile.
  std::optional<Timings> timings_;

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
  using IncludeInDumpsStore = FixedSizeValueStore<SemIR::CheckIRId, bool>;
  using TreeAndSubtreesGettersStore = Parse::GetTreeAndSubtreesStore;

  // This relies on construction after `units` are all initialized, which is
  // reflected by the `ArrayRef` here.
  explicit MultiUnitCache(
      const CompileOptions* options,
      llvm::ArrayRef<std::unique_ptr<CompilationUnit>> units)
      : options_(options), units_(units) {}

  // If `include_in_dumps` is in use, we need to apply per-file include
  // settings.
  auto ApplyPerFileIncludeInDumps() -> void {
    if (!include_in_dumps_) {
      // No cached value to update.
      return;
    }
    for (const auto& [i, unit] : llvm::enumerate(units_)) {
      if (unit->has_include_in_dumps()) {
        include_in_dumps_->Set(SemIR::CheckIRId(i), true);
      }
    }
  }

  auto include_in_dumps() -> const IncludeInDumpsStore& {
    if (!include_in_dumps_) {
      include_in_dumps_.emplace(
          IncludeInDumpsStore::MakeWithExplicitSize(units_.size(), false));
      for (const auto& [i, unit] : llvm::enumerate(units_)) {
        // If this is first accessed after lexing is complete, we need to apply
        // per-file includes. Otherwise, this is based only on the exclude
        // option.
        bool include =
            unit->has_include_in_dumps() ||
            llvm::none_of(options_->exclude_dump_file_prefixes,
                          [&](auto prefix) {
                            return unit->input_filename().starts_with(prefix);
                          });
        include_in_dumps_->Set(SemIR::CheckIRId(i), include);
      }
    }
    return *include_in_dumps_;
  }

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
  const CompileOptions* options_;

  // The units being compiled.
  llvm::ArrayRef<std::unique_ptr<CompilationUnit>> units_;

  // For each unit, whether it's included in dumps. Used cross-phase.
  std::optional<IncludeInDumpsStore> include_in_dumps_;

  // For each unit, the `TreeAndSubtrees` getter. Used by lowering.
  std::optional<TreeAndSubtreesGettersStore> tree_and_subtrees_getters_;
};

}  // namespace

CompilationUnit::CompilationUnit(SemIR::CheckIRId check_ir_id,
                                 int total_ir_count, DriverEnv* driver_env,
                                 const CompileOptions* options,
                                 Diagnostics::Consumer* consumer,
                                 llvm::StringRef input_filename,
                                 const llvm::Target* target)
    : check_ir_id_(check_ir_id),
      total_ir_count_(total_ir_count),
      driver_env_(driver_env),
      options_(options),
      target_(target),
      input_filename_(input_filename),
      vlog_stream_(driver_env_->vlog_stream) {
  if (vlog_stream_ != nullptr || options_->stream_errors) {
    consumer_ = consumer;
  } else {
    sorting_consumer_ = Diagnostics::SortingConsumer(*consumer);
    consumer_ = &*sorting_consumer_;
  }
}

auto CompilationUnit::IncludeInDumps() -> bool {
  return cache_->include_in_dumps().Get(check_ir_id_);
}

auto CompilationUnit::SetMultiUnitCache(MultiUnitCache* cache) -> void {
  CARBON_CHECK(!cache_, "Called SetMultiUnitCache twice");
  cache_ = cache;

  if (options_->dump_mem_usage && IncludeInDumps()) {
    CARBON_CHECK(!mem_usage_);
    mem_usage_ = MemUsage();
  }
  if (options_->dump_timings && IncludeInDumps()) {
    CARBON_CHECK(!timings_);
    timings_ = Timings();
  }
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

  if (mem_usage_) {
    mem_usage_->Add("source_", source_->text().size(), source_->text().size());
  }

  CARBON_VLOG("*** SourceBuffer ***\n```\n{0}\n```\n", source_->text());

  LogCall("Lex::Lex", "lex", [&] {
    Lex::LexOptions options;
    options.consumer = consumer_;
    options.vlog_stream = vlog_stream_;
    if (options_->dump_tokens && IncludeInDumps()) {
      options.dump_stream = driver_env_->output_stream;
      options.omit_file_boundary_tokens = options_->omit_file_boundary_tokens;
    }
    tokens_ = Lex::Lex(value_stores_, *source_, options);
  });
  if (mem_usage_) {
    mem_usage_->Collect("tokens_", *tokens_);
  }
  if (tokens_->has_errors()) {
    success_ = false;
  }
}

auto CompilationUnit::RunParse() -> void {
  LogCall("Parse::Parse", "parse", [&] {
    Parse::ParseOptions options;
    options.consumer = consumer_;
    options.vlog_stream = vlog_stream_;
    if (options_->dump_parse_tree && IncludeInDumps()) {
      options.dump_stream = driver_env_->output_stream;
      options.dump_preorder_parse_tree = options_->preorder_parse_tree;
    }
    parse_tree_ = Parse::Parse(*tokens_, options);
  });
  if (mem_usage_) {
    mem_usage_->Collect("parse_tree_", *parse_tree_);
  }
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
          .timings = timings_ ? &*timings_ : nullptr,
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

  if (mem_usage_) {
    mem_usage_->Collect("sem_ir_", *sem_ir_);
  }

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
    options.llvm_verifier_stream =
        options_->run_llvm_verifier ? driver_env_->error_stream : nullptr;
    options.want_debug_info = options_->include_debug_info;
    options.vlog_stream = vlog_stream_;
    options.opt_level = options_->opt_level;
    options.mangle_string_fingerprint = options_->mangle_string_fingerprint;
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
  llvm::Triple target_triple(options_->codegen_options.target);
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
  bool opt_for_speed = options_->opt_level == Lower::OptimizationLevel::Speed;
  bool opt_for_size_or_speed =
      opt_for_speed || options_->opt_level == Lower::OptimizationLevel::Size;
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
      CompileOptions::GetLLVMOptimizationLevel(options_->opt_level));

  if (vlog_stream_) {
    CARBON_VLOG("*** Running pass pipeline: ");
    pass_manager.printPipeline(
        *vlog_stream_, [&pic](llvm::StringRef class_name) {
          auto pass_name = pic.getPassNameForClassName(class_name);
          return pass_name.empty() ? class_name : pass_name;
        });
    CARBON_VLOG(" ***\n");
  }

  LogCall("ModulePassManager::run", "optimize",
          [&] { pass_manager.run(*module_, mam); });

  if (vlog_stream_) {
    CARBON_VLOG("*** Optimized llvm::Module ***\n");
    module_->print(*vlog_stream_, /*AAW=*/nullptr,
                   /*ShouldPreserveUseListOrder=*/false,
                   /*IsForDebug=*/true);
  }
}

auto CompilationUnit::PostLower() -> void {
  CARBON_CHECK(module_, "Must call RunLower first");
  if (options_->dump_llvm_ir && IncludeInDumps()) {
    module_->print(*driver_env_->output_stream, /*AAW=*/nullptr,
                   /*ShouldPreserveUseListOrder=*/true);
  }
}

auto CompilationUnit::RunCodeGen() -> void {
  CARBON_CHECK(module_, "Must call RunLower first");
  LogCall("CodeGen", "codegen", [&] { success_ = RunCodeGenHelper(); });
}

auto CompilationUnit::PostCompile() -> void {
  if (options_->dump_shared_values && IncludeInDumps()) {
    Yaml::Print(*driver_env_->output_stream,
                value_stores_.OutputYaml(input_filename_));
  }
  if (mem_usage_) {
    mem_usage_->Collect("value_stores_", value_stores_);
    Yaml::Print(*driver_env_->output_stream,
                mem_usage_->OutputYaml(input_filename_));
  }
  if (timings_) {
    Yaml::Print(*driver_env_->output_stream,
                timings_->OutputYaml(input_filename_));
  }

  // The diagnostics consumer must be flushed before compilation artifacts are
  // destructed, because diagnostics can refer to their state.
  consumer_->Flush();
}

auto CompilationUnit::RunCodeGenHelper() -> bool {
  CARBON_CHECK(module_, "Must call RunLower first");
  CARBON_CHECK(target_machine_, "Must call MakeTargetMachine first");

  CodeGen codegen(module_.get(), target_machine_.get(), consumer_);
  if (vlog_stream_) {
    CARBON_VLOG("*** Assembly ***\n");
    codegen.EmitAssembly(*vlog_stream_);
  }

  if (options_->output_filename == "-") {
    // TODO: The output file name, forcing object output, and requesting
    // textual assembly output are all somewhat linked flags. We should add
    // some validation that they are used correctly.
    if (options_->force_obj_output) {
      if (!codegen.EmitObject(*driver_env_->output_stream)) {
        return false;
      }
    } else {
      if (!codegen.EmitAssembly(*driver_env_->output_stream)) {
        return false;
      }
    }
  } else {
    llvm::SmallString<256> output_filename = options_->output_filename;
    if (output_filename.empty()) {
      if (!source_->is_regular_file()) {
        // Don't invent file names like `-.o` or `/dev/stdin.o`.
        // TODO: Consider rephrasing the diagnostic to use the file as the
        // `Emit` location.
        CARBON_DIAGNOSTIC(CompileInputNotRegularFile, Error,
                          "output file name must be specified for input `{0}` "
                          "that is not a regular file",
                          std::string);
        driver_env_->emitter.Emit(CompileInputNotRegularFile, input_filename_);
        return false;
      }
      output_filename = input_filename_;
      llvm::sys::path::replace_extension(output_filename,
                                         options_->asm_output ? ".s" : ".o");
    }
    CARBON_VLOG("Writing output to: {0}\n", output_filename);

    std::error_code ec;
    llvm::raw_fd_ostream output_file(output_filename, ec,
                                     llvm::sys::fs::OF_None);
    if (ec) {
      // TODO: Consider rephrasing the diagnostic to use the file as the `Emit`
      // location.
      CARBON_DIAGNOSTIC(CompileOutputFileOpenError, Error,
                        "could not open output file `{0}`: {1}", std::string,
                        std::string);
      driver_env_->emitter.Emit(CompileOutputFileOpenError,
                                output_filename.str().str(), ec.message());
      return false;
    }
    if (options_->asm_output) {
      if (!codegen.EmitAssembly(output_file)) {
        return false;
      }
    } else {
      if (!codegen.EmitObject(output_file)) {
        return false;
      }
    }
  }
  return true;
}

auto CompilationUnit::GetParseTreeAndSubtrees()
    -> const Parse::TreeAndSubtrees& {
  if (!parse_tree_and_subtrees_) {
    parse_tree_and_subtrees_ = Parse::TreeAndSubtrees(*tokens_, *parse_tree_);
    if (mem_usage_) {
      mem_usage_->Collect("parse_tree_and_subtrees_",
                          *parse_tree_and_subtrees_);
    }
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
  Timings::ScopedTiming timing(timings_ ? &*timings_ : nullptr, timing_label);
  fn();
  CARBON_VLOG("*** {0} done ***\n", logging_label);
}

auto CompileSubcommand::Run(DriverEnv& driver_env) -> DriverResult {
  if (!ValidateOptions(driver_env.emitter)) {
    return {.success = false};
  }

  // Validate the target before passing it to Clang.
  const llvm::Target* target;
  if (auto t = options_.ValidateTarget(driver_env.emitter); !t.ok()) {
    return {.success = false};
  } else {
    target = *t;
  }

  std::shared_ptr<clang::CompilerInvocation> clang_invocation;
  {
    if (driver_env.fuzzing && !options_.clang_args.empty()) {
      // Parsing specific Clang arguments can reach deep into
      // external libraries that aren't fuzz clean.
      TestAndDiagnoseIfFuzzingExternalLibraries(driver_env, "compile");
      return {.success = false};
    }

    if (auto i = options_.BuildClangInvocation(driver_env); !i.ok()) {
      return {.success = false};
    } else {
      clang_invocation = *i;
    }
  }

  // Find the files comprising the prelude if we are importing it.
  // TODO: Replace this with a search for library api files in a
  // package-specific search path based on the library name.
  llvm::SmallVector<std::string> prelude;
  if (options_.prelude_import && !options_.custom_core &&
      options_.phase >= CompileOptions::Phase::Check) {
    if (auto find = driver_env.installation->ReadPreludeManifest();
        !find.ok()) {
      // TODO: Change ReadPreludeManifest to produce diagnostics.
      CARBON_DIAGNOSTIC(CompilePreludeManifestError, Error, "{0}", std::string);
      driver_env.emitter.Emit(CompilePreludeManifestError,
                              PrintToString(find.error()));
      return {.success = false};
    } else {
      prelude = std::move(*find);
    }
  }

  // Prepare CompilationUnits before building scope exit handlers.
  llvm::SmallVector<std::unique_ptr<CompilationUnit>> units;
  int unit_index = -1;
  int total_unit_count = prelude.size() + options_.input_filenames.size();
  auto unit_builder = [&](llvm::StringRef filename) {
    ++unit_index;
    return std::make_unique<CompilationUnit>(
        SemIR::CheckIRId(unit_index), total_unit_count, &driver_env, &options_,
        &driver_env.consumer, filename, target);
  };
  llvm::append_range(units, llvm::map_range(prelude, unit_builder));
  llvm::append_range(units,
                     llvm::map_range(options_.input_filenames, unit_builder));
  CARBON_CHECK(units.size() == static_cast<size_t>(total_unit_count));

  // Add the cache to all units. This must be done after all units are created.
  MultiUnitCache cache(&options_, units);
  for (auto& unit : units) {
    unit->SetMultiUnitCache(&cache);
  }

  auto on_exit = llvm::scope_exit([&]() {
    // Finish compilation units. This flushes their diagnostics in the order in
    // which they were specified on the command line.
    for (auto& unit : units) {
      unit->PostCompile();
    }

    driver_env.consumer.Flush();
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
      driver_env.consumer.set_stream(&out);
    }

    for (auto& unit : units) {
      unit->FlushForStackTrace();
    }
    driver_env.consumer.Flush();
    driver_env.consumer.set_stream(driver_env.error_stream);
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
  cache.ApplyPerFileIncludeInDumps();
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

  // Gather Check::Units.
  llvm::SmallVector<Check::Unit> check_units;
  check_units.reserve(units.size());
  for (auto& unit : units) {
    if (unit->has_source()) {
      check_units.push_back(unit->GetCheckUnit());
    }
  }

  // Execute the actual checking.
  CARBON_VLOG_TO(driver_env.vlog_stream, "*** Check::CheckParseTrees ***\n");
  Check::CheckParseTreesOptions options;
  options.prelude_import = options_.prelude_import;
  options.vlog_stream = driver_env.vlog_stream;
  options.fuzzing = driver_env.fuzzing;
  options.mangle_string_fingerprint = options_.mangle_string_fingerprint;
  if (options.vlog_stream || options_.dump_sem_ir || options_.dump_cpp_ast ||
      options_.dump_raw_sem_ir) {
    options.include_in_dumps = &cache.include_in_dumps();
    if (options_.dump_sem_ir) {
      options.dump_stream = driver_env.output_stream;
    }
    if (options_.dump_cpp_ast) {
      options.dump_cpp_ast_stream = driver_env.output_stream;
    }
    if (options.vlog_stream || options_.dump_sem_ir) {
      options.dump_sem_ir_ranges = options_.dump_sem_ir_ranges;
    }
    if (options_.dump_raw_sem_ir) {
      options.raw_dump_stream = driver_env.output_stream;
      options.dump_raw_sem_ir_builtins = options_.builtin_sem_ir;
    }
    options.sem_ir_crash_dump = options_.sem_ir_crash_dump;
  }
  Check::CheckParseTrees(check_units, cache.tree_and_subtrees_getters(),
                         driver_env.fs, options, clang_invocation);
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
  if (llvm::any_of(units, [&](const auto& unit) { return !unit->success(); })) {
    CARBON_VLOG_TO(driver_env.vlog_stream,
                   "*** Stopping before lowering due to errors ***\n");
    return make_result();
  }

  // Lower and optimize.
  for (const auto& unit : units) {
    unit->RunLower();

    if (options_.phase != CompileOptions::Phase::Lower) {
      unit->RunOptimize(*clang_invocation);
    }

    unit->PostLower();
  }
  if (options_.phase == CompileOptions::Phase::Lower ||
      options_.phase == CompileOptions::Phase::Optimize) {
    return make_result();
  }
  CARBON_CHECK(options_.phase == CompileOptions::Phase::CodeGen,
               "CodeGen should be the last stage");

  bool output_last_input_only = options_.output_last_input_only;
  if (!output_last_input_only && units.size() > 1 &&
      !options_.output_filename.empty() && options_.output_filename != "-") {
    // TODO: Command line structure should change to make this implicit (passing
    // non-compiling inputs differently), and the warning should be removed.
    CARBON_DIAGNOSTIC(
        CompileMultipleInputsWithOutput, Warning,
        "only outputting {0} to {1}, skipping output of {2} input "
        "file{2:s}; pass `--output-last-input-only` to silence this warning",
        std::string, std::string, Diagnostics::IntAsSelect);
    driver_env.emitter.Emit(CompileMultipleInputsWithOutput,
                            units.back()->input_filename().str(),
                            options_.output_filename.str(), units.size() - 1);
    output_last_input_only = true;
  }

  // Codegen.
  if (output_last_input_only) {
    units.back()->RunCodeGen();
  } else {
    for (const auto& unit : units) {
      unit->RunCodeGen();
    }
  }
  return make_result();
}

}  // namespace Carbon
