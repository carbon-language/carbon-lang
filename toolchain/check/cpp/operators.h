// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_OPERATORS_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_OPERATORS_H_

#include "toolchain/check/context.h"
#include "toolchain/check/operator.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Looks up the given operator in the Clang AST generated when importing C++
// code using argument dependent lookup (ADL) and return overload set
// instruction.
auto LookupCppOperator(Context& context, SemIR::LocId loc_id, Operator op,
                       llvm::ArrayRef<SemIR::InstId> arg_ids) -> SemIR::InstId;

// Looks up the given operator in the Clang AST generated when importing C++
// code using argument dependent lookup (ADL) and return overload set
// instruction.
//
// This overload synthesises objects in an unevaluated context within the Clang
// AST based on the types it is provided.
auto LookupCppOperator(Context& context, SemIR::LocId loc_id, Operator op,
                       llvm::ArrayRef<SemIR::TypeId> arg_type_ids)
    -> SemIR::InstId;

// Returns whether the decl is an operator member function.
auto IsCppOperatorMethodDecl(clang::Decl* decl) -> bool;

// Returns whether the specified instruction refers to a C++ overloaded operator
// that is a method. If so, the first operand will be passed as `self` rather
// than as the first argument.
auto IsCppOperatorMethod(Context& context, SemIR::InstId inst_id) -> bool;

// Returns whether the specified instruction refers to a C++ constructor or
// non-operator method. If so, when mapping from a Carbon interface to a C++
// call, we pass a `self` parameter as the first argument instead.
auto IsCppConstructorOrNonMethodOperator(Context& context,
                                         SemIR::InstId inst_id) -> bool;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_OPERATORS_H_
