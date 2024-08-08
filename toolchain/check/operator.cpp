// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/operator.h"

#include "toolchain/check/call.h"
#include "toolchain/check/context.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/member_access.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Returns the name scope of the operator interface for the specified operator
// from the Core package.
static auto GetOperatorInterface(Context& context, SemIR::LocId loc_id,
                                 Operator op) -> SemIR::NameScopeId {
  auto interface_id = context.LookupNameInCore(loc_id, op.interface_name);
  if (interface_id == SemIR::InstId::BuiltinError) {
    return SemIR::NameScopeId::Invalid;
  }

  // We expect it to be an interface.
  if (auto interface_inst =
          context.insts().TryGetAs<SemIR::InterfaceType>(interface_id)) {
    return context.interfaces().Get(interface_inst->interface_id).scope_id;
  }

  CARBON_DIAGNOSTIC(CoreNameNotInterface, Error,
                    "Expected name `Core.{0}` implicitly referenced here to "
                    "name an interface.",
                    llvm::StringLiteral);
  CARBON_DIAGNOSTIC(CoreNameNotInterfacePrevious, Note,
                    "Name declared here.");
  context.emitter()
      .Build(loc_id, CoreNameNotInterface, op.interface_name)
      .Note(interface_id, CoreNameNotInterfacePrevious)
      .Emit();
  return SemIR::NameScopeId::Invalid;
}

// Returns the `Op` function for the specified operator.
static auto GetOperatorOpFunction(Context& context, SemIR::LocId loc_id,
                                  Operator op) -> SemIR::InstId {
  auto interface_scope_id = GetOperatorInterface(context, loc_id, op);
  if (!interface_scope_id.is_valid()) {
    return SemIR::InstId::BuiltinError;
  }

  // TODO: For a parameterized interface, find the corresponding specific.
  // TODO: Require the interface to be a complete type.
  LookupScope scope = {.name_scope_id = interface_scope_id,
                       .specific_id = SemIR::SpecificId::Invalid};

  // Lookup `Interface.Op`.
  auto op_ident_id = context.identifiers().Add(op.op_name);
  auto op_result = context.LookupQualifiedName(
      loc_id, SemIR::NameId::ForIdentifier(op_ident_id), scope,
      /*required=*/false);
  if (op_result.inst_id.is_valid()) {
    // Look through import_refs and aliases.
    auto op_const_id = GetConstantValueInSpecific(
        context.sem_ir(), op_result.specific_id, op_result.inst_id);
    auto op_id = context.constant_values().GetInstId(op_const_id);

    // We expect it to be an associated function.
    if (context.insts().Is<SemIR::AssociatedEntity>(op_id)) {
      return op_id;
    }
  }

  CARBON_DIAGNOSTIC(
      CoreNameNotAssociatedFunction, Error,
      "Expected name `Core.{0}.{1}` implicitly referenced here to "
      "name an associated function.",
      llvm::StringLiteral, llvm::StringLiteral);
  CARBON_DIAGNOSTIC(CoreNameNotAssociatedFunctionPrevious, Note,
                    "Name declared here.");
  auto emitter = context.emitter().Build(loc_id, CoreNameNotAssociatedFunction,
                                         op.interface_name, op.op_name);
  if (op_result.inst_id.is_valid()) {
    emitter.Note(op_result.inst_id, CoreNameNotAssociatedFunctionPrevious);
  }
  emitter.Emit();
  return SemIR::InstId::BuiltinError;
}

auto BuildUnaryOperator(Context& context, SemIR::LocId loc_id, Operator op,
                        SemIR::InstId operand_id,
                        BadOperatorDiagnoser diagnoser) -> SemIR::InstId {
                          (void)diagnoser;
  // Look up the operator function.
  auto op_fn = GetOperatorOpFunction(context, loc_id, op);

  // Form `operand.(Op)`.
  auto bound_op_id =
      PerformCompoundMemberAccess(context, loc_id, operand_id, op_fn);
  if (bound_op_id == SemIR::InstId::BuiltinError) {
    return SemIR::InstId::BuiltinError;
  }

  // Form `bound_op()`.
  return PerformCall(context, loc_id, bound_op_id, {});
}

auto BuildBinaryOperator(Context& context, SemIR::LocId loc_id, Operator op,
                         SemIR::InstId lhs_id, SemIR::InstId rhs_id,
                         BadOperatorDiagnoser diagnoser) -> SemIR::InstId {
                          (void)diagnoser;
  // Look up the operator function.
  auto op_fn = GetOperatorOpFunction(context, loc_id, op);

  // Form `lhs.(Op)`.
  auto bound_op_id =
      PerformCompoundMemberAccess(context, loc_id, lhs_id, op_fn);
  if (bound_op_id == SemIR::InstId::BuiltinError) {
    return SemIR::InstId::BuiltinError;
  }

  // Form `bound_op(rhs)`.
  return PerformCall(context, loc_id, bound_op_id, {rhs_id});
}

}  // namespace Carbon::Check
