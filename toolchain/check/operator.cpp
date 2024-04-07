// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/operator.h"

#include "toolchain/check/call.h"
#include "toolchain/check/context.h"
#include "toolchain/check/member_access.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Returns the name scope of the operator interface for the specified operator
// from the Core package.
static auto GetOperatorInterface(Context& context, Parse::AnyExprId node_id,
                                 Operator op) -> SemIR::NameScopeId {
  auto interface_id = context.LookupNameInCore(node_id, op.interface_name);
  if (interface_id == SemIR::InstId::BuiltinError) {
    return SemIR::NameScopeId::Invalid;
  }

  // We expect it to be an interface.
  if (auto interface_inst =
          context.insts().TryGetAs<SemIR::InterfaceType>(interface_id)) {
    return context.interfaces().Get(interface_inst->interface_id).scope_id;
  }
  return SemIR::NameScopeId::Invalid;
}

// Returns the `Op` function for the specified operator.
static auto GetOperatorOpFunction(Context& context, Parse::AnyExprId node_id,
                                  Operator op) -> SemIR::InstId {
  auto interface_scope_id = GetOperatorInterface(context, node_id, op);
  if (!interface_scope_id.is_valid()) {
    return SemIR::InstId::Invalid;
  }

  // Lookup `Interface.Op`.
  auto op_ident_id = context.identifiers().Add(op.op_name);
  auto op_id = context.LookupQualifiedName(
      node_id, SemIR::NameId::ForIdentifier(op_ident_id), interface_scope_id,
      /*required=*/false);
  if (!op_id.is_valid()) {
    return SemIR::InstId::Invalid;
  }

  // Look through import_refs and aliases.
  op_id = context.constant_values().Get(op_id).inst_id();

  // We expect it to be an associated function.
  if (context.insts().Is<SemIR::AssociatedEntity>(op_id)) {
    return op_id;
  }
  return SemIR::InstId::Invalid;
}

auto BuildUnaryOperator(Context& context, Parse::AnyExprId node_id, Operator op,
                        SemIR::InstId operand_id) -> SemIR::InstId {
  auto op_fn = GetOperatorOpFunction(context, node_id, op);
  if (!op_fn.is_valid()) {
    context.TODO(node_id,
                 "missing or invalid operator interface, also avoid duplicate "
                 "diagnostic if prelude is unavailable");
    return SemIR::InstId::BuiltinError;
  }

  // Form `operand.(Op)`.
  auto bound_op_id =
      PerformCompoundMemberAccess(context, node_id, operand_id, op_fn);
  if (bound_op_id == SemIR::InstId::BuiltinError) {
    return SemIR::InstId::BuiltinError;
  }

  // Form `bound_op()`.
  return PerformCall(context, node_id, bound_op_id, {});
}

auto BuildBinaryOperator(Context& context, Parse::AnyExprId node_id,
                         Operator op, SemIR::InstId lhs_id,
                         SemIR::InstId rhs_id) -> SemIR::InstId {
  auto op_fn = GetOperatorOpFunction(context, node_id, op);
  if (!op_fn.is_valid()) {
    context.TODO(node_id, "missing or invalid operator interface");
    return SemIR::InstId::BuiltinError;
  }

  // Form `lhs.(Op)`.
  auto bound_op_id =
      PerformCompoundMemberAccess(context, node_id, lhs_id, op_fn);
  if (bound_op_id == SemIR::InstId::BuiltinError) {
    return SemIR::InstId::BuiltinError;
  }

  // Form `bound_op(rhs)`.
  return PerformCall(context, node_id, bound_op_id, {rhs_id});
}

}  // namespace Carbon::Check
