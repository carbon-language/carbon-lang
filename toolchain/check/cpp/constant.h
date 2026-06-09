// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_CONSTANT_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_CONSTANT_H_

#include <optional>

#include "clang/AST/APValue.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace clang {
class VarDecl;
}  // namespace clang

namespace Carbon::Check {

// Converts an `APValue` to a Carbon `ConstantId`.
auto MapAPValueToConstant(Context& context, SemIR::LocId loc_id,
                          const clang::APValue& ap_value, clang::QualType type,
                          bool is_lvalue) -> SemIR::ConstantId;

// Converts a Carbon constant instruction to an `APValue`.
auto MapConstantToAPValue(Context& context, SemIR::InstId const_inst_id,
                          clang::QualType param_type)
    -> std::optional<clang::APValue>;

// Attempt to evaluate a C++ constexpr variable as a Carbon constant.
auto EvalCppVarDecl(Context& context, SemIR::LocId loc_id,
                    const clang::VarDecl* var_decl, SemIR::TypeId type_id)
    -> SemIR::ConstantId;

// Attempt to evaluate a call to a C++ constexpr/consteval function as a
// Carbon constant.
auto EvalCppCall(Context& context, SemIR::LocId loc_id,
                 const SemIR::ClangDecl& clang_decl, SemIR::InstBlockId args_id)
    -> SemIR::ConstantId;

// If the callee is a C++ thunk, modify `call` to directly call the
// callee of the C++ thunk. Otherwise, does nothing and leaves `call`
// unmodified.
auto MaybeModifyCppThunkCallForConstEval(Context& context, SemIR::Call* call)
    -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_CONSTANT_H_
