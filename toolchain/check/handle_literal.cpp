// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleLiteral(Context& context, Parse::Node parse_node) -> bool {
  auto token = context.parse_tree().node_token(parse_node);
  switch (auto token_kind = context.tokens().GetKind(token)) {
    case Lex::TokenKind::False:
    case Lex::TokenKind::True: {
      context.AddNodeAndPush(
          parse_node,
          SemIR::Node::BoolLiteral::Make(
              parse_node,
              context.CanonicalizeType(SemIR::NodeId::BuiltinBoolType),
              token_kind == Lex::TokenKind::True ? SemIR::BoolValue::True
                                                 : SemIR::BoolValue::False));
      break;
    }
    case Lex::TokenKind::IntegerLiteral: {
      auto id = context.semantics_ir().AddIntegerLiteral(
          context.tokens().GetIntegerLiteral(token));
      context.AddNodeAndPush(
          parse_node,
          SemIR::Node::IntegerLiteral::Make(
              parse_node,
              context.CanonicalizeType(SemIR::NodeId::BuiltinIntegerType), id));
      break;
    }
    case Lex::TokenKind::RealLiteral: {
      auto token_value = context.tokens().GetRealLiteral(token);
      auto id = context.semantics_ir().AddRealLiteral(
          {.mantissa = token_value.mantissa,
           .exponent = token_value.exponent,
           .is_decimal = token_value.is_decimal});
      context.AddNodeAndPush(
          parse_node,
          SemIR::Node::RealLiteral::Make(
              parse_node,
              context.CanonicalizeType(SemIR::NodeId::BuiltinFloatingPointType),
              id));
      break;
    }
    case Lex::TokenKind::StringLiteral: {
      auto id = context.semantics_ir().AddString(
          context.tokens().GetStringLiteral(token));
      context.AddNodeAndPush(
          parse_node,
          SemIR::Node::StringLiteral::Make(
              parse_node,
              context.CanonicalizeType(SemIR::NodeId::BuiltinStringType), id));
      break;
    }
    case Lex::TokenKind::Type: {
      context.node_stack().Push(parse_node, SemIR::NodeId::BuiltinTypeType);
      break;
    }
    case Lex::TokenKind::Bool: {
      context.node_stack().Push(parse_node, SemIR::NodeId::BuiltinBoolType);
      break;
    }
    case Lex::TokenKind::IntegerTypeLiteral: {
      auto text = context.tokens().GetTokenText(token);
      if (text != "i32") {
        return context.TODO(parse_node, "Currently only i32 is allowed");
      }
      context.node_stack().Push(parse_node, SemIR::NodeId::BuiltinIntegerType);
      break;
    }
    case Lex::TokenKind::FloatingPointTypeLiteral: {
      auto text = context.tokens().GetTokenText(token);
      if (text != "f64") {
        return context.TODO(parse_node, "Currently only f64 is allowed");
      }
      context.node_stack().Push(parse_node,
                                SemIR::NodeId::BuiltinFloatingPointType);
      break;
    }
    case Lex::TokenKind::StringTypeLiteral: {
      context.node_stack().Push(parse_node, SemIR::NodeId::BuiltinStringType);
      break;
    }
    default: {
      return context.TODO(parse_node, llvm::formatv("Handle {0}", token_kind));
    }
  }

  return true;
}

}  // namespace Carbon::Check
