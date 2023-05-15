// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"

namespace Carbon {

auto SemanticsHandleAddress(SemanticsContext& context,
                            ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleAddress");
}

auto SemanticsHandleBreakStatement(SemanticsContext& context,
                                   ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleBreakStatement");
}

auto SemanticsHandleBreakStatementStart(SemanticsContext& context,
                                        ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleBreakStatementStart");
}

auto SemanticsHandleCodeBlock(SemanticsContext& context,
                              ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleCodeBlock");
}

auto SemanticsHandleCodeBlockStart(SemanticsContext& context,
                                   ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleCodeBlockStart");
}

auto SemanticsHandleContinueStatement(SemanticsContext& context,
                                      ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleContinueStatement");
}

auto SemanticsHandleContinueStatementStart(SemanticsContext& context,
                                           ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleContinueStatementStart");
}

auto SemanticsHandleDeclaredName(SemanticsContext& context,
                                 ParseTree::Node parse_node) -> bool {
  // The parent is responsible for binding the name.
  context.node_stack().Push(parse_node);
  return true;
}

auto SemanticsHandleDeducedParameterList(SemanticsContext& context,
                                         ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleDeducedParameterList");
}

auto SemanticsHandleDeducedParameterListStart(SemanticsContext& context,
                                              ParseTree::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleDeducedParameterListStart");
}

auto SemanticsHandleDesignatedName(SemanticsContext& context,
                                   ParseTree::Node parse_node) -> bool {
  auto name_str = context.parse_tree().GetNodeText(parse_node);
  auto name_id = context.semantics().AddString(name_str);
  // The parent is responsible for binding the name.
  context.node_stack().Push(parse_node, name_id);
  return true;
}

auto SemanticsHandleDesignatorExpression(SemanticsContext& context,
                                         ParseTree::Node parse_node) -> bool {
  auto [_, name_id] = context.node_stack().PopForParseNodeAndNameId(
      ParseNodeKind::DesignatedName);

  auto base_id = context.node_stack().PopForNodeId();
  auto base = context.semantics().GetNode(base_id);
  auto base_type = context.semantics().GetNode(base.type_id());

  switch (base_type.kind()) {
    case SemanticsNodeKind::StructType: {
      auto refs =
          context.semantics().GetNodeBlock(base_type.GetAsStructType().second);
      // TODO: Do we need to optimize this with a lookup table for O(1)?
      for (int i = 0; i < static_cast<int>(refs.size()); ++i) {
        auto ref = context.semantics().GetNode(refs[i]);
        if (name_id == ref.GetAsStructTypeField()) {
          context.AddNodeAndPush(
              parse_node,
              SemanticsNode::StructMemberAccess::Make(
                  parse_node, ref.type_id(), base_id, SemanticsMemberIndex(i)));
          return true;
        }
      }
      CARBON_DIAGNOSTIC(DesignatorExpressionNameNotFound, Error,
                        "Type `{0}` does not have a member `{1}`.", std::string,
                        llvm::StringRef);
      context.emitter().Emit(parse_node, DesignatorExpressionNameNotFound,
                             context.semantics().StringifyNode(base.type_id()),
                             context.semantics().GetString(name_id));
      break;
    }
    default: {
      CARBON_DIAGNOSTIC(DesignatorExpressionUnsupported, Error,
                        "Type `{0}` does not support designator expressions.",
                        std::string);
      context.emitter().Emit(parse_node, DesignatorExpressionUnsupported,
                             context.semantics().StringifyNode(base.type_id()));
      break;
    }
  }

  // Should only be reached on error.
  context.node_stack().Push(parse_node, SemanticsNodeId::BuiltinInvalidType);
  return true;
}

auto SemanticsHandleEmptyDeclaration(SemanticsContext& context,
                                     ParseTree::Node parse_node) -> bool {
  // Empty declarations have no actions associated, but we still balance the
  // tree.
  context.node_stack().Push(parse_node);
  return true;
}

auto SemanticsHandleExpressionStatement(SemanticsContext& context,
                                        ParseTree::Node parse_node) -> bool {
  // Pop the expression without investigating its contents.
  // TODO: This will probably eventually need to do some "do not discard"
  // analysis.
  context.node_stack().PopAndDiscardId();
  context.node_stack().Push(parse_node);
  return true;
}

auto SemanticsHandleFileEnd(SemanticsContext& /*context*/,
                            ParseTree::Node /*parse_node*/) -> bool {
  // Do nothing, no need to balance this node.
  return true;
}

auto SemanticsHandleForHeader(SemanticsContext& context,
                              ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleForHeader");
}

auto SemanticsHandleForHeaderStart(SemanticsContext& context,
                                   ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleForHeaderStart");
}

auto SemanticsHandleForIn(SemanticsContext& context, ParseTree::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleForIn");
}

auto SemanticsHandleForStatement(SemanticsContext& context,
                                 ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleForStatement");
}

auto SemanticsHandleGenericPatternBinding(SemanticsContext& context,
                                          ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "GenericPatternBinding");
}

auto SemanticsHandleIfCondition(SemanticsContext& context,
                                ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleIfCondition");
}

auto SemanticsHandleIfConditionStart(SemanticsContext& context,
                                     ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleIfConditionStart");
}

auto SemanticsHandleIfStatement(SemanticsContext& context,
                                ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleIfStatement");
}

auto SemanticsHandleIfStatementElse(SemanticsContext& context,
                                    ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleIfStatementElse");
}

auto SemanticsHandleInfixOperator(SemanticsContext& context,
                                  ParseTree::Node parse_node) -> bool {
  auto rhs_id = context.node_stack().PopForNodeId();
  auto lhs_id = context.node_stack().PopForNodeId();

  // TODO: This should search for a compatible interface. For now, it's a very
  // trivial check of validity on the operation.
  lhs_id = context.ImplicitAsRequired(
      parse_node, lhs_id, context.semantics().GetNode(rhs_id).type_id());

  // Figure out the operator for the token.
  auto token = context.parse_tree().node_token(parse_node);
  switch (auto token_kind = context.tokens().GetKind(token)) {
    case TokenKind::Plus:
      context.AddNodeAndPush(
          parse_node,
          SemanticsNode::BinaryOperatorAdd::Make(
              parse_node, context.semantics().GetNode(lhs_id).type_id(), lhs_id,
              rhs_id));
      break;
    default:
      return context.TODO(parse_node, llvm::formatv("Handle {0}", token_kind));
  }

  return true;
}

auto SemanticsHandleInvalidParse(SemanticsContext& context,
                                 ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleInvalidParse");
}

auto SemanticsHandleLiteral(SemanticsContext& context,
                            ParseTree::Node parse_node) -> bool {
  auto token = context.parse_tree().node_token(parse_node);
  switch (auto token_kind = context.tokens().GetKind(token)) {
    case TokenKind::IntegerLiteral: {
      auto id = context.semantics().AddIntegerLiteral(
          context.tokens().GetIntegerLiteral(token));
      context.AddNodeAndPush(
          parse_node, SemanticsNode::IntegerLiteral::Make(parse_node, id));
      break;
    }
    case TokenKind::RealLiteral: {
      auto token_value = context.tokens().GetRealLiteral(token);
      auto id = context.semantics().AddRealLiteral(
          {.mantissa = token_value.Mantissa(),
           .exponent = token_value.Exponent(),
           .is_decimal = token_value.IsDecimal()});
      context.AddNodeAndPush(parse_node,
                             SemanticsNode::RealLiteral::Make(parse_node, id));
      break;
    }
    case TokenKind::StringLiteral: {
      auto id = context.semantics().AddString(
          context.tokens().GetStringLiteral(token));
      context.AddNodeAndPush(
          parse_node, SemanticsNode::StringLiteral::Make(parse_node, id));
      break;
    }
    case TokenKind::IntegerTypeLiteral: {
      auto text = context.tokens().GetTokenText(token);
      if (text != "i32") {
        return context.TODO(parse_node, "Currently only i32 is allowed");
      }
      context.node_stack().Push(parse_node,
                                SemanticsNodeId::BuiltinIntegerType);
      break;
    }
    case TokenKind::FloatingPointTypeLiteral: {
      auto text = context.tokens().GetTokenText(token);
      if (text != "f64") {
        return context.TODO(parse_node, "Currently only f64 is allowed");
      }
      context.node_stack().Push(parse_node,
                                SemanticsNodeId::BuiltinFloatingPointType);
      break;
    }
    case TokenKind::StringTypeLiteral: {
      context.node_stack().Push(parse_node, SemanticsNodeId::BuiltinStringType);
      break;
    }
    default: {
      return context.TODO(parse_node, llvm::formatv("Handle {0}", token_kind));
    }
  }

  return true;
}

auto SemanticsHandleNameReference(SemanticsContext& context,
                                  ParseTree::Node parse_node) -> bool {
  auto name = context.parse_tree().GetNodeText(parse_node);
  context.node_stack().Push(parse_node, context.LookupName(parse_node, name));
  return true;
}

auto SemanticsHandleNamedConstraintDeclaration(SemanticsContext& context,
                                               ParseTree::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleNamedConstraintDeclaration");
}

auto SemanticsHandleNamedConstraintDefinition(SemanticsContext& context,
                                              ParseTree::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleNamedConstraintDefinition");
}

auto SemanticsHandleNamedConstraintDefinitionStart(SemanticsContext& context,
                                                   ParseTree::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleNamedConstraintDefinitionStart");
}

auto SemanticsHandleNamedConstraintIntroducer(SemanticsContext& context,
                                              ParseTree::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleNamedConstraintIntroducer");
}

auto SemanticsHandleParameterList(SemanticsContext& context,
                                  ParseTree::Node parse_node) -> bool {
  auto [ir_id, refs_id] = context.ParamOrArgEnd(
      /*for_args=*/false, ParseNodeKind::ParameterListStart);

  context.PopScope();
  context.node_stack().PopAndDiscardSoloParseNode(
      ParseNodeKind::ParameterListStart);
  context.finished_params_stack().push_back({ir_id, refs_id});
  context.node_stack().Push(parse_node);
  return true;
}

auto SemanticsHandleParameterListComma(SemanticsContext& context,
                                       ParseTree::Node /*parse_node*/) -> bool {
  context.ParamOrArgComma(/*for_args=*/false);
  return true;
}

auto SemanticsHandleParameterListStart(SemanticsContext& context,
                                       ParseTree::Node parse_node) -> bool {
  context.PushScope();
  context.node_stack().Push(parse_node);
  context.ParamOrArgStart();
  return true;
}

auto SemanticsHandleParenExpression(SemanticsContext& context,
                                    ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleParenExpression");
}

auto SemanticsHandleParenExpressionOrTupleLiteralStart(
    SemanticsContext& context, ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleParenExpressionOrTupleLiteralStart");
}

auto SemanticsHandlePatternBinding(SemanticsContext& context,
                                   ParseTree::Node parse_node) -> bool {
  auto [type_node, parsed_type_id] =
      context.node_stack().PopForParseNodeAndNodeId();
  SemanticsNodeId cast_type_id = context.ImplicitAsRequired(
      type_node, parsed_type_id, SemanticsNodeId::BuiltinTypeType);

  // Get the name.
  auto name_node = context.node_stack().PopForSoloParseNode();

  // Allocate storage, linked to the name for error locations.
  auto storage_id =
      context.AddNode(SemanticsNode::VarStorage::Make(name_node, cast_type_id));

  // Bind the name to storage.
  auto name_id = context.BindName(name_node, cast_type_id, storage_id);

  // If this node's result is used, it'll be for either the name or the
  // storage address. The storage address can be found through the name, so we
  // push the name.
  context.node_stack().Push(parse_node, name_id);

  return true;
}

auto SemanticsHandlePostfixOperator(SemanticsContext& context,
                                    ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandlePostfixOperator");
}

auto SemanticsHandlePrefixOperator(SemanticsContext& context,
                                   ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandlePrefixOperator");
}

auto SemanticsHandleReturnStatement(SemanticsContext& context,
                                    ParseTree::Node parse_node) -> bool {
  CARBON_CHECK(!context.return_scope_stack().empty());
  const auto& fn_node =
      context.semantics().GetNode(context.return_scope_stack().back());
  const auto callable = context.semantics().GetCallable(
      fn_node.GetAsFunctionDeclaration().second);

  if (context.parse_tree().node_kind(context.node_stack().PeekParseNode()) ==
      ParseNodeKind::ReturnStatementStart) {
    context.node_stack().PopAndDiscardSoloParseNode(
        ParseNodeKind::ReturnStatementStart);

    if (callable.return_type_id.is_valid()) {
      // TODO: Add a note pointing at the return type's parse node.
      CARBON_DIAGNOSTIC(ReturnStatementMissingExpression, Error,
                        "Must return a {0}.", std::string);
      context.emitter()
          .Build(parse_node, ReturnStatementMissingExpression,
                 context.semantics().StringifyNode(callable.return_type_id))
          .Emit();
    }

    context.AddNodeAndPush(parse_node, SemanticsNode::Return::Make(parse_node));
  } else {
    auto arg = context.node_stack().PopForNodeId();
    context.node_stack().PopAndDiscardSoloParseNode(
        ParseNodeKind::ReturnStatementStart);

    if (!callable.return_type_id.is_valid()) {
      CARBON_DIAGNOSTIC(
          ReturnStatementDisallowExpression, Error,
          "No return expression should be provided in this context.");
      CARBON_DIAGNOSTIC(ReturnStatementImplicitNote, Note,
                        "There was no return type provided.");
      context.emitter()
          .Build(parse_node, ReturnStatementDisallowExpression)
          .Note(fn_node.parse_node(), ReturnStatementImplicitNote)
          .Emit();
    } else {
      arg =
          context.ImplicitAsRequired(parse_node, arg, callable.return_type_id);
    }

    context.AddNodeAndPush(
        parse_node,
        SemanticsNode::ReturnExpression::Make(
            parse_node, context.semantics().GetNode(arg).type_id(), arg));
  }
  return true;
}

auto SemanticsHandleReturnStatementStart(SemanticsContext& context,
                                         ParseTree::Node parse_node) -> bool {
  // No action, just a bracketing node.
  context.node_stack().Push(parse_node);
  return true;
}

auto SemanticsHandleReturnType(SemanticsContext& context,
                               ParseTree::Node parse_node) -> bool {
  // Propagate the type expression.
  auto [type_parse_node, type_node_id] =
      context.node_stack().PopForParseNodeAndNodeId();
  auto cast_node_id = context.ImplicitAsRequired(
      type_parse_node, type_node_id, SemanticsNodeId::BuiltinTypeType);
  context.node_stack().Push(parse_node, cast_node_id);
  return true;
}

auto SemanticsHandleSelfTypeIdentifier(SemanticsContext& context,
                                       ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleSelfTypeIdentifier");
}

auto SemanticsHandleSelfValueIdentifier(SemanticsContext& context,
                                        ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleSelfValueIdentifier");
}

auto SemanticsHandleTemplate(SemanticsContext& context,
                             ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleTemplate");
}

auto SemanticsHandleTupleLiteral(SemanticsContext& context,
                                 ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleTupleLiteral");
}

auto SemanticsHandleTupleLiteralComma(SemanticsContext& context,
                                      ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleTupleLiteralComma");
}

auto SemanticsHandleVariableDeclaration(SemanticsContext& context,
                                        ParseTree::Node parse_node) -> bool {
  auto [last_parse_node, last_node_id] =
      context.node_stack().PopForParseNodeAndNodeId();

  if (context.parse_tree().node_kind(last_parse_node) !=
      ParseNodeKind::PatternBinding) {
    auto storage_id =
        context.node_stack().PopForNodeId(ParseNodeKind::VariableInitializer);

    auto binding = context.node_stack().PopForParseNodeAndNameId(
        ParseNodeKind::PatternBinding);

    // Restore the name now that the initializer is complete.
    context.ReaddNameToLookup(binding.second, storage_id);

    auto cast_value_id = context.ImplicitAsRequired(
        parse_node, last_node_id,
        context.semantics().GetNode(storage_id).type_id());
    context.AddNode(SemanticsNode::Assign::Make(
        parse_node, context.semantics().GetNode(cast_value_id).type_id(),
        storage_id, cast_value_id));
  }

  context.node_stack().PopAndDiscardSoloParseNode(
      ParseNodeKind::VariableIntroducer);
  context.node_stack().Push(parse_node);

  return true;
}

auto SemanticsHandleVariableIntroducer(SemanticsContext& context,
                                       ParseTree::Node parse_node) -> bool {
  // No action, just a bracketing node.
  context.node_stack().Push(parse_node);
  return true;
}

auto SemanticsHandleVariableInitializer(SemanticsContext& context,
                                        ParseTree::Node parse_node) -> bool {
  auto storage_id = context.TempRemoveLatestNameFromLookup();
  context.node_stack().Push(parse_node, storage_id);
  return true;
}

auto SemanticsHandleWhileCondition(SemanticsContext& context,
                                   ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleWhileCondition");
}

auto SemanticsHandleWhileConditionStart(SemanticsContext& context,
                                        ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleWhileConditionStart");
}

auto SemanticsHandleWhileStatement(SemanticsContext& context,
                                   ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleWhileStatement");
}

}  // namespace Carbon
