// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/member_access.h"
#include "toolchain/check/name_component.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/pointer_dereference.h"
#include "toolchain/check/type.h"
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
    // TODO: Pass in the expression we're trying to dereference to produce a
    // better diagnostic.
    CARBON_DIAGNOSTIC(ArrowOperatorOfNonPointer, Error,
                      "cannot apply `->` operator to non-pointer type {0}",
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

// Returns the `NameId` for an identifier node.
static auto GetIdentifierAsNameId(
    Context& context, Parse::NodeIdOneOf<Parse::IdentifierNameNotBeforeParamsId,
                                         Parse::IdentifierNameBeforeParamsId,
                                         Parse::IdentifierNameExprId>
                          node_id) -> SemIR::NameId {
  CARBON_CHECK(!context.parse_tree().node_has_error(node_id),
               "TODO: Support checking error parse nodes");
  auto token = context.parse_tree().node_token(node_id);
  return SemIR::NameId::ForIdentifier(context.tokens().GetIdentifier(token));
}

// Handle a name that is used as an expression by performing unqualified name
// lookup.
static auto HandleNameAsExpr(Context& context, Parse::NodeId node_id,
                             SemIR::NameId name_id) -> SemIR::InstId {
  auto result = LookupUnqualifiedName(context, node_id, name_id);
  SemIR::InstId inst_id = result.scope_result.target_inst_id();
  auto value = context.insts().Get(inst_id);
  auto type_id = SemIR::GetTypeInSpecific(context.sem_ir(), result.specific_id,
                                          value.type_id());
  CARBON_CHECK(type_id.has_value(), "Missing type for {0}", value);

  // If the named entity has a constant value that depends on its specific,
  // store the specific too.
  if (result.specific_id.has_value() &&
      context.constant_values().Get(inst_id).is_symbolic()) {
    inst_id =
        AddInst<SemIR::SpecificConstant>(context, node_id,
                                         {.type_id = type_id,
                                          .inst_id = inst_id,
                                          .specific_id = result.specific_id});
  }

  return AddInst<SemIR::NameRef>(
      context, node_id,
      {.type_id = type_id, .name_id = name_id, .value_id = inst_id});
}

auto HandleParseNode(Context& context,
                     Parse::IdentifierNameNotBeforeParamsId node_id) -> bool {
  // The parent is responsible for binding the name.
  context.node_stack().Push(node_id, GetIdentifierAsNameId(context, node_id));
  return true;
}

auto HandleParseNode(Context& context,
                     Parse::IdentifierNameBeforeParamsId node_id) -> bool {
  // Push a pattern block stack entry to handle the parameter pattern.
  context.pattern_block_stack().Push();
  context.full_pattern_stack().PushFullPattern(
      FullPatternStack::Kind::ImplicitParamList);
  // The parent is responsible for binding the name.
  context.node_stack().Push(node_id, GetIdentifierAsNameId(context, node_id));
  return true;
}

auto HandleParseNode(Context& context, Parse::IdentifierNameExprId node_id)
    -> bool {
  auto name_id = GetIdentifierAsNameId(context, node_id);
  context.node_stack().Push(node_id,
                            HandleNameAsExpr(context, node_id, name_id));
  return true;
}

// Returns the `NameId` for a keyword node.
static auto GetKeywordAsNameId(
    Context& context, Parse::NodeIdOneOf<Parse::KeywordNameNotBeforeParamsId,
                                         Parse::KeywordNameBeforeParamsId>
                          node_id) -> SemIR::NameId {
  auto token = context.parse_tree().node_token(node_id);
  switch (auto token_kind = context.tokens().GetKind(token)) {
    case Lex::TokenKind::Destroy:
      return SemIR::NameId::Destroy;
    default:
      CARBON_FATAL("Unexpected token kind: {0}", token_kind);
  }
}

auto HandleParseNode(Context& context,
                     Parse::KeywordNameNotBeforeParamsId node_id) -> bool {
  // The parent is responsible for binding the name.
  context.node_stack().Push(node_id, GetKeywordAsNameId(context, node_id));
  return true;
}

auto HandleParseNode(Context& context, Parse::KeywordNameBeforeParamsId node_id)
    -> bool {
  // Push a pattern block stack entry to handle the parameter pattern.
  context.pattern_block_stack().Push();
  context.full_pattern_stack().PushFullPattern(
      FullPatternStack::Kind::ImplicitParamList);
  // The parent is responsible for binding the name.
  context.node_stack().Push(node_id, GetKeywordAsNameId(context, node_id));
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
  context.node_stack().Push(
      node_id, HandleNameAsExpr(context, node_id, SemIR::NameId::SelfType));
  return true;
}

auto HandleParseNode(Context& context, Parse::SelfValueNameId node_id) -> bool {
  context.node_stack().Push(node_id, SemIR::NameId::SelfValue);
  return true;
}

auto HandleParseNode(Context& context, Parse::SelfValueNameExprId node_id)
    -> bool {
  context.node_stack().Push(
      node_id, HandleNameAsExpr(context, node_id, SemIR::NameId::SelfValue));
  return true;
}

// Common logic for name qualifiers.
static auto ApplyNameQualifier(Context& context) -> bool {
  context.decl_name_stack().ApplyNameQualifier(PopNameComponent(context));
  return true;
}

auto HandleParseNode(Context& context,
                     Parse::IdentifierNameQualifierWithParamsId /*node_id*/)
    -> bool {
  return ApplyNameQualifier(context);
}

auto HandleParseNode(Context& context,
                     Parse::IdentifierNameQualifierWithoutParamsId /*node_id*/)
    -> bool {
  return ApplyNameQualifier(context);
}

auto HandleParseNode(Context& context,
                     Parse::KeywordNameQualifierWithParamsId /*node_id*/)
    -> bool {
  return ApplyNameQualifier(context);
}

auto HandleParseNode(Context& context,
                     Parse::KeywordNameQualifierWithoutParamsId /*node_id*/)
    -> bool {
  return ApplyNameQualifier(context);
}

auto HandleParseNode(Context& context, Parse::DesignatorExprId node_id)
    -> bool {
  SemIR::NameId name_id = context.node_stack().PopName();

  if (name_id == SemIR::NameId::SelfType) {
    // Look up `.Self`.
    SemIR::InstId period_self_id =
        HandleNameAsExpr(context, node_id, SemIR::NameId::PeriodSelf);
    context.node_stack().Push(node_id, period_self_id);
  } else {
    // Otherwise this is `.Member`, so look up `.Self` and then `Member` in
    // `.Self`.
    SemIR::InstId period_self_id = SemIR::InstId::None;
    {
      // TODO: Instead of annotating the diagnostic, should change
      // `HandleNameAsExpr` to optionally allow us to produce the diagnostic
      // instead so we can generate a "name `.Self` implicitly referenced by
      // designated expression, but not found" diagnostic instead of adding a
      // note to the current "name `.Self` not found" message.
      DiagnosticAnnotationScope annotate_diagnostics(
          &context.emitter(), [&](auto& builder) {
            CARBON_DIAGNOSTIC(
                NoPeriodSelfForDesignator, Note,
                "designator may only be used when `.Self` is in scope");
            builder.Note(SemIR::LocId::None, NoPeriodSelfForDesignator);
          });
      period_self_id =
          HandleNameAsExpr(context, node_id, SemIR::NameId::PeriodSelf);
    }
    auto member_id =
        PerformMemberAccess(context, node_id, period_self_id, name_id);
    context.node_stack().Push(node_id, member_id);
  }
  return true;
}

auto HandleParseNode(Context& context, Parse::PackageExprId node_id) -> bool {
  AddInstAndPush<SemIR::NameRef>(
      context, node_id,
      {.type_id =
           GetSingletonType(context, SemIR::NamespaceType::SingletonInstId),
       .name_id = SemIR::NameId::PackageNamespace,
       .value_id = SemIR::Namespace::PackageInstId});
  return true;
}

auto HandleParseNode(Context& context, Parse::CoreNameExprId node_id) -> bool {
  // TODO: Unqualified lookup will never find anything; perform lookup directly
  // into file scope.
  context.node_stack().Push(
      node_id, HandleNameAsExpr(context, node_id, SemIR::NameId::Core));
  return true;
}

}  // namespace Carbon::Check
