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
  llvm::StringLiteral op_name = "Op";
};

// Checks and builds SemIR for a unary operator expression. For example,
// `$operand` or `operand$`.
auto BuildUnaryOperator(Context& context, Parse::AnyExprId node_id, Operator op,
                        SemIR::InstId operand_id) -> SemIR::InstId;

// Checks and builds SemIR for a binary operator expression. For example,
// `lhs_id $ rhs_id`.
auto BuildBinaryOperator(Context& context, Parse::AnyExprId node_id,
                         Operator op, SemIR::InstId lhs_id,
                         SemIR::InstId rhs_id) -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_OPERATOR_H_
