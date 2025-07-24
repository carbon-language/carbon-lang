// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CHECK_H_
#define CARBON_TOOLCHAIN_CHECK_CHECK_H_

#include "clang/Frontend/CompilerInvocation.h"
#include "common/ostream.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/base/timings.h"
#include "toolchain/check/diagnostic_emitter.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Checking information that's tracked per file. All members are caller-owned.
// Other than `timings`, members must be non-null.
struct Unit {
  Diagnostics::Consumer* consumer;
  SharedValueStores* value_stores;
  // The `timings` may be null if nothing is to be recorded.
  Timings* timings;

  // The unit's SemIR, provided as empty and filled in by CheckParseTrees.
  SemIR::File* sem_ir;

  // Storage for the unit's Clang AST. The unique_ptr should start empty, and
  // can be assigned as part of checking.
  std::unique_ptr<clang::ASTUnit>* cpp_ast;
};

struct CheckParseTreesOptions {
  // Options must be set individually, not through initialization.
  explicit CheckParseTreesOptions() = default;

  // Whether to import the prelude.
  bool prelude_import = false;

  // If set, enables verbose output.
  llvm::raw_ostream* vlog_stream = nullptr;

  // Whether fuzzing is being run. Used to disable features we don't want to
  // fuzz.
  bool fuzzing = false;

  // Whether to include each unit in dumps. This is required when dumping
  // (either of `dump_stream` or `raw_dump_stream`), and must have entries based
  // on CheckIRId.
  const FixedSizeValueStore<SemIR::CheckIRId, bool>* include_in_dumps = nullptr;

  // If set, SemIR will be dumped to this.
  llvm::raw_ostream* dump_stream = nullptr;

  // When dumping textual SemIR (or printing it to for verbose output), whether
  // to use ranges.
  enum class DumpSemIRRanges : int8_t {
    IfPresent,
    Only,
    Ignore,
  };
  DumpSemIRRanges dump_sem_ir_ranges = DumpSemIRRanges::IfPresent;

  // If set, raw SemIR will be dumped to this.
  llvm::raw_ostream* raw_dump_stream = nullptr;

  // When dumping raw SemIR, whether to include builtins.
  bool dump_raw_sem_ir_builtins = false;
};

// Checks a group of parse trees. This will use imports to decide the order of
// checking.
//
// `units` will only contain units which should be checked, and is not indexed
// by `CheckIRId`.
auto CheckParseTrees(
    llvm::MutableArrayRef<Unit> units,
    const Parse::GetTreeAndSubtreesStore& tree_and_subtrees_getters,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
    const CheckParseTreesOptions& options,
    std::shared_ptr<clang::CompilerInvocation> clang_invocation) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CHECK_H_
