// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/member_access.h"
#include "toolchain/check/name_component.h"
#include "toolchain/check/pointer_dereference.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleMemberAccessExpr(Context& context, Parse::MemberAccessExprId node_id)
    -> bool {
  if (context.node_stack().PeekIs<Parse::NodeKind::ParenExpr>()) {
    auto member_expr_id = context.node_stack().PopExpr();
    auto base_id = context.node_stack().PopExpr();
    auto member_id =
        PerformCompoundMemberAccess(context, node_id, base_id, member_expr_id);
    context.node_stack().Push(node_id, member_id);
  } else {
    SemIR::NameId name_id = context.node_stack().PopName();
    auto base_id = context.node_stack().PopExpr();
    auto member_id = PerformMemberAccess(context, node_id, base_id, name_id);
    context.node_stack().Push(node_id, member_id);
  }
  return true;
}

auto HandlePointerMemberAccessExpr(Context& context,
                                   Parse::PointerMemberAccessExprId node_id)
    -> bool {
  auto diagnose_not_pointer = [&context,
                               &node_id](SemIR::TypeId not_pointer_type_id) {
    CARBON_DIAGNOSTIC(ArrowOperatorOfNonPointer, Error,
                      "Cannot apply `->` operator to non-pointer type `{0}`.",
                      SemIR::TypeId);

    auto builder = context.emitter().Build(
        TokenOnly(node_id), ArrowOperatorOfNonPointer, not_pointer_type_id);
    builder.Emit();
  };

  if (context.node_stack().PeekIs<Parse::NodeKind::ParenExpr>()) {
    auto member_expr_id = context.node_stack().PopExpr();
    auto base_id = context.node_stack().PopExpr();
    auto deref_base_id = PerformPointerDereference(context, node_id, base_id,
                                                   diagnose_not_pointer);
    auto member_id = PerformCompoundMemberAccess(context, node_id,
                                                 deref_base_id, member_expr_id);
    context.node_stack().Push(node_id, member_id);
  } else {
    SemIR::NameId name_id = context.node_stack().PopName();
    auto base_id = context.node_stack().PopExpr();
    auto deref_base_id = PerformPointerDereference(context, node_id, base_id,
                                                   diagnose_not_pointer);
    auto member_id =
        PerformMemberAccess(context, node_id, deref_base_id, name_id);
    context.node_stack().Push(node_id, member_id);
  }

  return true;
}

static auto GetIdentifierAsName(Context& context, Parse::NodeId node_id)
    -> std::optional<SemIR::NameId> {
  auto token = context.parse_tree().node_token(node_id);
  if (context.tokens().GetKind(token) != Lex::TokenKind::Identifier) {
    CARBON_CHECK(context.parse_tree().node_has_error(node_id));
    return std::nullopt;
  }
  return SemIR::NameId::ForIdentifier(context.tokens().GetIdentifier(token));
}

// Handle a name that is used as an expression by performing unqualified name
// lookup.
static auto HandleNameAsExpr(Context& context, Parse::NodeId node_id,
                             SemIR::NameId name_id) -> bool {
  auto value_id = context.LookupUnqualifiedName(node_id, name_id);
  auto value = context.insts().Get(value_id);
  context.AddInstAndPush<SemIR::NameRef>(
      node_id,
      {.type_id = value.type_id(), .name_id = name_id, .value_id = value_id});
  return true;
}

auto HandleIdentifierName(Context& context, Parse::IdentifierNameId node_id)
    -> bool {
  // The parent is responsible for binding the name.
  auto name_id = GetIdentifierAsName(context, node_id);
  if (!name_id) {
    return context.TODO(node_id, "Error recovery from keyword name.");
  }
  context.node_stack().Push(node_id, *name_id);
  return true;
}

auto HandleIdentifierNameExpr(Context& context,
                              Parse::IdentifierNameExprId node_id) -> bool {
  auto name_id = GetIdentifierAsName(context, node_id);
  if (!name_id) {
    return context.TODO(node_id, "Error recovery from keyword name.");
  }
  return HandleNameAsExpr(context, node_id, *name_id);
}

auto HandleBaseName(Context& context, Parse::BaseNameId node_id) -> bool {
  context.node_stack().Push(node_id, SemIR::NameId::Base);
  return true;
}

auto HandleSelfTypeNameExpr(Context& context, Parse::SelfTypeNameExprId node_id)
    -> bool {
  return HandleNameAsExpr(context, node_id, SemIR::NameId::SelfType);
}

auto HandleSelfValueName(Context& context, Parse::SelfValueNameId node_id)
    -> bool {
  context.node_stack().Push(node_id, SemIR::NameId::SelfValue);
  return true;
}

auto HandleSelfValueNameExpr(Context& context,
                             Parse::SelfValueNameExprId node_id) -> bool {
  return HandleNameAsExpr(context, node_id, SemIR::NameId::SelfValue);
}

auto HandleNameQualifier(Context& context, Parse::NameQualifierId /*node_id*/)
    -> bool {
  context.decl_name_stack().ApplyNameQualifier(PopNameComponent(context));
  return true;
}

auto HandlePackageExpr(Context& context, Parse::PackageExprId node_id) -> bool {
  context.AddInstAndPush<SemIR::NameRef>(
      node_id,
      {.type_id = context.GetBuiltinType(SemIR::BuiltinKind::NamespaceType),
       .name_id = SemIR::NameId::PackageNamespace,
       .value_id = SemIR::InstId::PackageNamespace});
  return true;
}

}  // namespace Carbon::Check
