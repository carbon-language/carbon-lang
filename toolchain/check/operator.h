// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_OPERATOR_H_
#define CARBON_TOOLCHAIN_CHECK_OPERATOR_H_

#include "toolchain/check/context.h"
#include "toolchain/check/core_identifier.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

struct Operator {
  CoreIdentifier interface_name;
  llvm::ArrayRef<SemIR::InstId> interface_args_ref = {};
  CoreIdentifier op_name = CoreIdentifier::Op;
};

// Checks and builds SemIR for a unary operator expression. For example,
// `*operand` or `operand*`.
//
// On failure, an ErrorInst is returned and a diagnostic is produced unless
// `diagnose` is false. It is incorrect to specify `diagnose` as false if the
// resulting ErrorInst may appear in the produced SemIR.
//
// If specified, `missing_impl_diagnostic_context` is used to provide context
// for the diagnostic if the impl lookup for the operator fails.
auto BuildUnaryOperator(Context& context, SemIR::LocId loc_id, Operator op,
                        SemIR::InstId operand_id, bool diagnose = true,
                        DiagnosticContextFn missing_impl_diagnostic_context =
                            nullptr) -> SemIR::InstId;

// Checks and builds SemIR for a binary operator expression. For example,
// `lhs_id * rhs_id`.
//
// // On failure, an ErrorInst is returned and a diagnostic is produced unless
// `diagnose` is false. It is incorrect to specify `diagnose` as false if the
// resulting ErrorInst may appear in the produced SemIR.
//
// If specified, `missing_impl_diagnostic_context` is used to provide context
// for the diagnostic if the impl lookup for the operator fails.
auto BuildBinaryOperator(
    Context& context, SemIR::LocId loc_id, Operator op, SemIR::InstId lhs_id,
    SemIR::InstId rhs_id, bool diagnose = true,
    DiagnosticContextFn missing_impl_diagnostic_context = nullptr)
    -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_OPERATOR_H_
