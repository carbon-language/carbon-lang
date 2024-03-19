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

// Returns the scope of the Core package, or Invalid if it's not found.
//
// TODO: Consider tracking the Core package in SemIR so we don't need to use
// name lookup to find it.
static auto GetCorePackage(Context& context, Parse::AnyExprId node_id)
    -> SemIR::NameScopeId {
  // TODO: If the current package is the `Core` package, return
  // `SemIR::InstId::Package`.

  auto ident_id = context.identifiers().Lookup("Core");
  if (!ident_id.is_valid()) {
    return SemIR::NameScopeId::Invalid;
  }
  auto name_id = SemIR::NameId::ForIdentifier(ident_id);

  // Look up `package.Core`.
  auto package_id = context.LookupQualifiedName(
      node_id, name_id, SemIR::NameScopeId::Package, /*required=*/false);
  if (!package_id.is_valid()) {
    return SemIR::NameScopeId::Invalid;
  }

  // Look through import_refs and aliases.
  package_id = context.constant_values().Get(package_id).inst_id();

  // We expect it to be a package, and fail if not.
  if (auto package_inst =
          context.insts().TryGetAs<SemIR::Namespace>(package_id)) {
    auto& name_scope = context.name_scopes().Get(package_inst->name_scope_id);
    // Check that this is really the `Core` package and not an alias.
    if (name_scope.is_closed_import && name_scope.name_id == name_id &&
        name_scope.enclosing_scope_id == SemIR::NameScopeId::Package) {
      return package_inst->name_scope_id;
    }
  }
  return SemIR::NameScopeId::Invalid;
}

// Returns the name scope of the operator interface for the specified operator
// from the Core package.
static auto GetOperatorInterface(Context& context, Parse::AnyExprId node_id,
                                 Operator op) -> SemIR::NameScopeId {
  auto carbon_package_id = GetCorePackage(context, node_id);
  if (!carbon_package_id.is_valid()) {
    return SemIR::NameScopeId::Invalid;
  }

  // Lookup `Core.InterfaceName`.
  auto interface_ident_id = context.identifiers().Add(op.interface_name);
  auto interface_id = context.LookupQualifiedName(
      node_id, SemIR::NameId::ForIdentifier(interface_ident_id),
      carbon_package_id, /*required=*/false);
  if (!interface_id.is_valid()) {
    return SemIR::NameScopeId::Invalid;
  }

  // Look through import_refs and aliases.
  interface_id = context.constant_values().Get(interface_id).inst_id();

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
    context.TODO(node_id, "missing or invalid operator interface");
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
