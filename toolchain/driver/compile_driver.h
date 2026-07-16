// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_COMPILE_DRIVER_H_
#define CARBON_TOOLCHAIN_DRIVER_COMPILE_DRIVER_H_

#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Target/TargetMachine.h"
#include "toolchain/diagnostics/sorting_consumer.h"
#include "toolchain/driver/compile_options.h"
#include "toolchain/driver/driver_env.h"

namespace Carbon {

class MultiUnitCache;

// Ties together information for a file being compiled.
class CompilationUnit {
 public:
  // `driver_env`, `options`, `consumer`, and `target` must be non-null.
  explicit CompilationUnit(SemIR::CheckIRId check_ir_id, int total_ir_count,
                           DriverEnv* driver_env, const CompileOptions* options,
                           Diagnostics::Consumer* consumer,
                           llvm::StringRef input_filename,
                           std::string output_filename,
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
  auto output_filename() -> llvm::StringRef { return output_filename_; }
  auto has_include_in_dumps() -> bool {
    return tokens_ && tokens_->has_include_in_dumps();
  }
  auto success() -> bool { return success_; }
  auto has_source() -> bool { return source_.has_value(); }
  auto get_trees_and_subtrees() -> Parse::GetTreeAndSubtreesFn {
    return *tree_and_subtrees_getter_;
  }

  auto source() const -> const SourceBuffer& { return *source_; }
  auto tokens() const -> const Lex::TokenizedBuffer& { return *tokens_; }
  auto parse_tree() const -> const Parse::Tree& { return *parse_tree_; }
  auto parse_tree_and_subtrees() const -> const Parse::TreeAndSubtrees& {
    return GetParseTreeAndSubtrees();
  }

 private:
  // Do codegen. Returns true on success.
  auto RunCodeGenHelper() -> bool;

  // The TreeAndSubtrees is mainly used for debugging and diagnostics, and has
  // significant overhead. Avoid constructing it when unused.
  auto GetParseTreeAndSubtrees() const -> const Parse::TreeAndSubtrees&;

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
  std::string output_filename_;

  // Copied from driver_ for CARBON_VLOG.
  llvm::raw_pwrite_stream* vlog_stream_;

  // Diagnostics are sent to consumer_, with optional sorting.
  std::optional<Diagnostics::SortingConsumer> sorting_consumer_;
  Diagnostics::Consumer* consumer_;

  bool success_ = true;

  // Initialized by `SetMultiUnitCache`.
  MultiUnitCache* cache_ = nullptr;
  // Tracks memory usage of the compile. Present when usage is being dumped or
  // collected into `DriverEnv::mem_usage`; see `SetMultiUnitCache`.
  mutable std::optional<MemUsage> mem_usage_;
  // Tracks timings of the compile.
  std::optional<Timings> timings_;

  // These are initialized as steps are run.
  std::optional<SourceBuffer> source_;
  std::optional<Lex::TokenizedBuffer> tokens_;
  std::optional<Parse::Tree> parse_tree_;
  mutable std::optional<Parse::TreeAndSubtrees> parse_tree_and_subtrees_;
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

// Helper class to compile C++ and Carbon input files. Used by the `build` and
// `compile` subcommands.
class CompileDriver {
 public:
  explicit CompileDriver(CompileOptions* options);

  // Configure the toolchain to compile all input files and dependencies.
  // The `map_input` function maps an input file name to an output static
  // object name.
  // Returns `false` on configuration error.
  auto Initialize(
      DriverEnv& driver_env,
      llvm::function_ref<auto(llvm::StringRef)->std::string> map_input) -> bool;

  // Performs the compilation process on each input specified in the
  // `CompileOptions` provided at construction time.
  auto Compile(DriverEnv& driver_env) -> DriverResult;

  // Returns the index in the `units()` array of the first input file
  // specified by the user on the command line. This may not be the first
  // element in the array due to dependency prepends.
  auto first_input_index() -> size_t { return input_filenames_index_; }

  // Provides read-only access to the array of CompilationUnits, useful for
  // subsequent processing of the compiled output products.
  auto units() -> llvm::ArrayRef<std::unique_ptr<CompilationUnit>> {
    return units_;
  }

 private:
  CompileOptions* options_;
  size_t input_filenames_index_ = 0;
  llvm::SmallVector<std::unique_ptr<CompilationUnit>, 256> units_;
  std::unique_ptr<MultiUnitCache> cache_;
  std::shared_ptr<clang::CompilerInvocation> clang_invocation_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_COMPILE_DRIVER_H_
