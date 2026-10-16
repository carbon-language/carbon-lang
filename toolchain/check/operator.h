// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_OPERATOR_H_
#define CARBON_TOOLCHAIN_CHECK_OPERATOR_H_

#include <optional>

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

// Labels the operand `inst_id` with its type, using `label`. An operand whose
// type is an error is not labeled: the diagnostic is downstream of one already
// reported, and `<error>` in a label adds nothing.
inline auto AttachOperandType(Context& context, DiagnosticBuilder& builder,
                              SemIR::InstId inst_id,
                              const Diagnostics::LabelBase<TypeOfInstId>& label)
    -> void {
  if (inst_id.has_value() &&
      context.insts().Get(inst_id).type_id() != SemIR::ErrorInst::TypeId) {
    builder.Attach(inst_id, label, inst_id);
  }
}

// How diagnostics from an operation's `impl` lookup are located and labeled.
// The caller supplies this because it has the operands and their syntax, and
// the diagnostic is emitted several layers down.
struct MissingImplDiagnostic {
  // Where to locate the message, when the syntax has a better location than
  // the whole expression: the operator of an infix expression, the index of an
  // indexing expression, or the operand of a conversion. Unset locates it at
  // the expression.
  std::optional<LocIdForDiagnostics> loc_id = std::nullopt;
  // Attaches labels with the operands' types. Their text follows the syntax:
  // an infix operator has a left and a right operand, and `a[i]` has an object
  // and an index.
  DiagnosticAnnotateFn annotate = nullptr;
};

// Checks and builds SemIR for a unary operator expression. For example,
// `*operand` or `operand*`.
//
// On failure, an ErrorInst is returned and a diagnostic is produced unless
// `diagnose` is false. It is incorrect to specify `diagnose` as false if the
// resulting ErrorInst may appear in the produced SemIR.
//
// If specified, `missing_impl_diagnostic_context` is used to provide context
// for the diagnostic if the impl lookup for the operator fails, and
// `missing_impl` places and labels that diagnostic.
auto BuildUnaryOperator(
    Context& context, SemIR::LocId loc_id, Operator op,
    SemIR::InstId operand_id, bool diagnose = true,
    DiagnosticContextFn missing_impl_diagnostic_context = nullptr,
    MissingImplDiagnostic missing_impl = {}) -> SemIR::InstId;

// Checks and builds SemIR for a binary operator expression. For example,
// `lhs_id * rhs_id`.
//
// On failure, an ErrorInst is returned and a diagnostic is produced unless
// `diagnose` is false. It is incorrect to specify `diagnose` as false if the
// resulting ErrorInst may appear in the produced SemIR.
//
// If specified, `missing_impl_diagnostic_context` is used to provide context
// for the diagnostic if the impl lookup for the operator fails, and
// `missing_impl` places and labels that diagnostic.
auto BuildBinaryOperator(
    Context& context, SemIR::LocId loc_id, Operator op, SemIR::InstId lhs_id,
    SemIR::InstId rhs_id, bool diagnose = true,
    DiagnosticContextFn missing_impl_diagnostic_context = nullptr,
    MissingImplDiagnostic missing_impl = {}) -> SemIR::InstId;

// Builds `<object>.(Destroy.SelfDestruct)()`, converting adapters to their
// underlying types in the process.
auto BuildSelfDestructCall(Context& context, SemIR::InstId object_id)
    -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_OPERATOR_H_
