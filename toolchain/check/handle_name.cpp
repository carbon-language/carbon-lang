// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/member_access.h"
#include "toolchain/check/name_component.h"
#include "toolchain/check/pointer_dereference.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::MemberAccessExprId node_id)
    -> bool {
  auto node_kind = context.node_stack().PeekNodeKind();

  if (node_kind == Parse::NodeKind::ParenExpr) {
    auto member_expr_id = context.node_stack().PopExpr();
    auto base_id = context.node_stack().PopExpr();
    auto member_id =
        PerformCompoundMemberAccess(context, node_id, base_id, member_expr_id);
    context.node_stack().Push(node_id, member_id);
  } else if (node_kind == Parse::NodeKind::IntLiteral) {
    auto index_inst_id = context.node_stack().PopExpr();
    auto tuple_inst_id = context.node_stack().PopExpr();

    auto tuple_value_inst_id =
        PerformTupleAccess(context, node_id, tuple_inst_id, index_inst_id);

    context.node_stack().Push(node_id, tuple_value_inst_id);
  } else {
    SemIR::NameId name_id = context.node_stack().PopName();
    auto base_id = context.node_stack().PopExpr();
    auto member_id = PerformMemberAccess(context, node_id, base_id, name_id);
    context.node_stack().Push(node_id, member_id);
  }
  return true;
}

auto HandleParseNode(Context& context, Parse::PointerMemberAccessExprId node_id)
    -> bool {
  auto diagnose_not_pointer = [&context,
                               &node_id](SemIR::TypeId not_pointer_type_id) {
    CARBON_DIAGNOSTIC(ArrowOperatorOfNonPointer, Error,
                      "cannot apply `->` operator to non-pointer type `{0}`",
                      SemIR::TypeId);

    auto builder = context.emitter().Build(
        TokenOnly(node_id), ArrowOperatorOfNonPointer, not_pointer_type_id);
    builder.Emit();
  };

  auto node_kind = context.node_stack().PeekNodeKind();

  if (node_kind == Parse::NodeKind::ParenExpr) {
    auto member_expr_id = context.node_stack().PopExpr();
    auto base_id = context.node_stack().PopExpr();
    auto deref_base_id = PerformPointerDereference(context, node_id, base_id,
                                                   diagnose_not_pointer);
    auto member_id = PerformCompoundMemberAccess(context, node_id,
                                                 deref_base_id, member_expr_id);
    context.node_stack().Push(node_id, member_id);
  } else if (node_kind == Parse::NodeKind::IntLiteral) {
    auto index_inst_id = context.node_stack().PopExpr();
    auto tuple_pointer_inst_id = context.node_stack().PopExpr();
    auto tuple_inst_id = PerformPointerDereference(
        context, node_id, tuple_pointer_inst_id, diagnose_not_pointer);
    auto tuple_value_inst_id =
        PerformTupleAccess(context, node_id, tuple_inst_id, index_inst_id);

    context.node_stack().Push(node_id, tuple_value_inst_id);
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

struct NodeAndInstIds {
  Parse::NodeId node_id;
  SemIR::InstId inst_id;
};

// Handle a name that is used as an expression by performing unqualified name
// lookup.
static auto HandleNameAsExpr(Context& context, Parse::NodeId node_id,
                             SemIR::NameId name_id) -> NodeAndInstIds {
  auto result = context.LookupUnqualifiedName(node_id, name_id);
  auto value = context.insts().Get(result.inst_id);
  auto type_id = SemIR::GetTypeInSpecific(context.sem_ir(), result.specific_id,
                                          value.type_id());
  CARBON_CHECK(type_id.is_valid(), "Missing type for {0}", value);

  // If the named entity has a constant value that depends on its specific,
  // store the specific too.
  if (result.specific_id.is_valid() &&
      context.constant_values().Get(result.inst_id).is_symbolic()) {
    result.inst_id = context.AddInst<SemIR::SpecificConstant>(
        node_id, {.type_id = type_id,
                  .inst_id = result.inst_id,
                  .specific_id = result.specific_id});
  }

  return {node_id, context.AddInst<SemIR::NameRef>(
                       node_id, {.type_id = type_id,
                                 .name_id = name_id,
                                 .value_id = result.inst_id})};
}

auto HandleParseNode(Context& context, Parse::IdentifierNameId node_id)
    -> bool {
  // The parent is responsible for binding the name.
  auto name_id = GetIdentifierAsName(context, node_id);
  if (!name_id) {
    return context.TODO(node_id, "Error recovery from keyword name.");
  }
  context.node_stack().Push(node_id, *name_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::IdentifierNameExprId node_id)
    -> bool {
  auto name_id = GetIdentifierAsName(context, node_id);
  if (!name_id) {
    return context.TODO(node_id, "Error recovery from keyword name.");
  }
  auto result = HandleNameAsExpr(context, node_id, *name_id);
  context.node_stack().Push(result.node_id, result.inst_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::BaseNameId node_id) -> bool {
  context.node_stack().Push(node_id, SemIR::NameId::Base);
  return true;
}

auto HandleParseNode(Context& context, Parse::SelfTypeNameId node_id) -> bool {
  context.node_stack().Push(node_id, SemIR::NameId::SelfType);
  return true;
}

auto HandleParseNode(Context& context, Parse::SelfTypeNameExprId node_id)
    -> bool {
  auto result = HandleNameAsExpr(context, node_id, SemIR::NameId::SelfType);
  context.node_stack().Push(result.node_id, result.inst_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::SelfValueNameId node_id) -> bool {
  context.node_stack().Push(node_id, SemIR::NameId::SelfValue);
  return true;
}

auto HandleParseNode(Context& context, Parse::SelfValueNameExprId node_id)
    -> bool {
  auto result = HandleNameAsExpr(context, node_id, SemIR::NameId::SelfValue);
  context.node_stack().Push(result.node_id, result.inst_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::NameQualifierId /*node_id*/)
    -> bool {
  context.decl_name_stack().ApplyNameQualifier(PopNameComponent(context));
  return true;
}

auto HandleParseNode(Context& context, Parse::DesignatorExprId node_id)
    -> bool {
  SemIR::NameId name_id = context.node_stack().PopName();

  // TODO: Give a clearer error when .Self is not found, particularly if this
  // is a `.Member` designator expression.
  SemIR::InstId period_self_id =
      HandleNameAsExpr(context, node_id, SemIR::NameId::PeriodSelf).inst_id;

  if (name_id == SemIR::NameId::SelfType) {
    // If this is `.Self`, result is the period_self_id we have computed.
    context.node_stack().Push(node_id, period_self_id);
  } else {
    // Otherwise this is `.Member`, so look up `Member` in `.Self`.
    // FIXME: Should we avoid generating a name reference to `.Self` in this
    // case?
    auto member_id =
        PerformMemberAccess(context, node_id, period_self_id, name_id);
    context.node_stack().Push(node_id, member_id);
  }
  return true;
}

auto HandleParseNode(Context& context, Parse::PackageExprId node_id) -> bool {
  context.AddInstAndPush<SemIR::NameRef>(
      node_id,
      {.type_id = context.GetBuiltinType(SemIR::BuiltinInstKind::NamespaceType),
       .name_id = SemIR::NameId::PackageNamespace,
       .value_id = SemIR::InstId::PackageNamespace});
  return true;
}

}  // namespace Carbon::Check
