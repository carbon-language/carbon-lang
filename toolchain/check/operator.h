// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_OPERATOR_H_
#define CARBON_TOOLCHAIN_CHECK_OPERATOR_H_

#include "toolchain/check/context.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

struct Operator {
  llvm::StringLiteral interface_name;
  llvm::ArrayRef<SemIR::InstId> interface_args_ref = {};
  llvm::StringLiteral op_name = "Op";
};

// Checks and builds SemIR for a unary operator expression. For example,
// `*operand` or `operand*`. If specified, `missing_impl_diagnoser` is used to
// build a custom error diagnostic for the case where impl lookup for the
// operator fails.
auto BuildUnaryOperator(Context& context, SemIR::LocId loc_id, Operator op,
                        SemIR::InstId operand_id,
                        Context::BuildDiagnosticFn missing_impl_diagnoser =
                            nullptr) -> SemIR::InstId;

// Checks and builds SemIR for a binary operator expression. For example,
// `lhs_id * rhs_id`. If specified, `missing_impl_diagnoser` is used to build a
// custom error diagnostic for the case where impl lookup for the operator
// fails.
auto BuildBinaryOperator(Context& context, SemIR::LocId loc_id, Operator op,
                         SemIR::InstId lhs_id, SemIR::InstId rhs_id,
                         Context::BuildDiagnosticFn missing_impl_diagnoser =
                             nullptr) -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_OPERATOR_H_
