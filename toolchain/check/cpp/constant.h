// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_CONSTANT_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_CONSTANT_H_

#include "clang/AST/APValue.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Converts an `APValue` to a Carbon `ConstantId`.
auto MapAPValueToConstant(Context& context, SemIR::LocId loc_id,
                          const clang::APValue& ap_value, clang::QualType type,
                          bool is_lvalue) -> SemIR::ConstantId;

// Attempt to evaluate a C++ constexpr variable as a Carbon constant.
auto EvalCppVarDecl(Context& context, SemIR::LocId loc_id,
                    const clang::VarDecl* var_decl, SemIR::TypeId type_id)
    -> SemIR::ConstantId;

// Attempt to evaluate a call to a C++ constexpr/consteval function as a
// Carbon constant.
auto EvalCppCall(Context& context, SemIR::LocId loc_id,
                 SemIR::ClangDeclId clang_decl_id, SemIR::InstBlockId args_id)
    -> SemIR::ConstantId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_CONSTANT_H_
