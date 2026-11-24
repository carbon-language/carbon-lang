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

// The return value of `MaybeModifyCalleeForConstEval`.
struct ModifiedCalleeArgs {
  // The backing storage for `args`, if a suitable array doesn't already exist.
  // Do not use this directly; it exists only to keep `args` from dangling.
  llvm::SmallVector<SemIR::InstId> arg_storage;

  // The updated argument list. This may be invalidated when `arg_storage` is
  // destroyed.
  llvm::ArrayRef<SemIR::InstId> args;
};

// If the callee is a C++ thunk, modify `call->callee_id` to point directly to
// the callee of the C++ thunk. Otherwise, does nothing and leaves `call`
// unmodified. This does not update `call->args_id`, because we may not be
// able to add new block IDs to `sem_ir`. Instead, this returns the updated
// argument list as an array.
auto MaybeModifyCalleeForConstEval(const SemIR::File& sem_ir, SemIR::Call* call)
    -> ModifiedCalleeArgs;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_CONSTANT_H_
