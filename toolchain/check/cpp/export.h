// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_EXPORT_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_EXPORT_H_

#include "clang/AST/Decl.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Get a `clang::FunctionDecl` that can be used to call a Carbon function.
auto GetReverseInteropFunctionDecl(Context& context, SemIR::LocId loc_id,
                                   clang::DeclContext& decl_context,
                                   SemIR::FunctionId function_id)
    -> clang::FunctionDecl*;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_EXPORT_H_
