// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CHECK_H_
#define CARBON_TOOLCHAIN_CHECK_CHECK_H_

#include "common/ostream.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/base/timings.h"
#include "toolchain/check/diagnostic_emitter.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/file.h"

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

// Checks a group of parse trees. This will use imports to decide the order of
// checking.
auto CheckParseTrees(
    llvm::MutableArrayRef<Unit> units,
    llvm::ArrayRef<Parse::GetTreeAndSubtreesFn> tree_and_subtrees_getters,
    bool prelude_import, llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
    llvm::StringRef target, llvm::raw_ostream* vlog_stream, bool fuzzing)
    -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CHECK_H_
