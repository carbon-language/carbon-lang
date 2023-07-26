// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"

namespace Carbon {

auto SemanticsHandleLiteral(SemanticsContext& context,
                            ParseTree::Node parse_node) -> bool {
  auto token = context.parse_tree().node_token(parse_node);
  switch (auto token_kind = context.tokens().GetKind(token)) {
    case TokenKind::False:
    case TokenKind::True: {
      context.AddNodeAndPush(
          parse_node,
          SemanticsNode::BoolLiteral::Make(
              parse_node,
              context.CanonicalizeType(SemanticsNodeId::BuiltinBoolType),
              token_kind == TokenKind::True ? SemanticsBoolValue::True
                                            : SemanticsBoolValue::False));
      break;
    }
    case TokenKind::IntegerLiteral: {
      auto id = context.semantics_ir().AddIntegerLiteral(
          context.tokens().GetIntegerLiteral(token));
      context.AddNodeAndPush(
          parse_node,
          SemanticsNode::IntegerLiteral::Make(
              parse_node,
              context.CanonicalizeType(SemanticsNodeId::BuiltinIntegerType),
              id));
      break;
    }
    case TokenKind::RealLiteral: {
      auto token_value = context.tokens().GetRealLiteral(token);
      auto id = context.semantics_ir().AddRealLiteral(
          {.mantissa = token_value.Mantissa(),
           .exponent = token_value.Exponent(),
           .is_decimal = token_value.IsDecimal()});
      context.AddNodeAndPush(parse_node,
                             SemanticsNode::RealLiteral::Make(
                                 parse_node,
                                 context.CanonicalizeType(
                                     SemanticsNodeId::BuiltinFloatingPointType),
                                 id));
      break;
    }
    case TokenKind::StringLiteral: {
      auto id = context.semantics_ir().AddString(
          context.tokens().GetStringLiteral(token));
      context.AddNodeAndPush(
          parse_node,
          SemanticsNode::StringLiteral::Make(
              parse_node,
              context.CanonicalizeType(SemanticsNodeId::BuiltinStringType),
              id));
      break;
    }
    case TokenKind::Type: {
      context.node_stack().Push(parse_node, SemanticsNodeId::BuiltinTypeType);
      break;
    }
    case TokenKind::Bool: {
      context.node_stack().Push(parse_node, SemanticsNodeId::BuiltinBoolType);
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

}  // namespace Carbon
