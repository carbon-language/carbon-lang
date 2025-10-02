// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_TYPE_MAPPING_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_TYPE_MAPPING_H_

#include "clang/AST/Type.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Invents a Clang argument expression to use in overload resolution to
// represent the given Carbon argument instruction.
auto InventClangArg(Context& context, SemIR::InstId arg_id) -> clang::Expr*;

// For each arg, invents a Clang argument expression to use in overload
// resolution or argument dependent lookup (ADL) to represent the given Carbon
// argument instructions. Returns std::nullopt if any arg failed.
auto InventClangArgs(Context& context, llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> std::optional<llvm::SmallVector<clang::Expr*>>;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_TYPE_MAPPING_H_
