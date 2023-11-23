// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleBoolLiteralFalse(Context& context, Parse::Node parse_node) -> bool {
  context.AddInstAndPush(
      parse_node,
      SemIR::BoolLiteral{parse_node,
                         context.GetBuiltinType(SemIR::BuiltinKind::BoolType),
                         SemIR::BoolValue::False});
  return true;
}

auto HandleBoolLiteralTrue(Context& context, Parse::Node parse_node) -> bool {
  context.AddInstAndPush(
      parse_node,
      SemIR::BoolLiteral{parse_node,
                         context.GetBuiltinType(SemIR::BuiltinKind::BoolType),
                         SemIR::BoolValue::True});
  return true;
}

auto HandleIntegerLiteral(Context& context, Parse::Node parse_node) -> bool {
  context.AddInstAndPush(
      parse_node,
      SemIR::IntegerLiteral{
          parse_node, context.GetBuiltinType(SemIR::BuiltinKind::IntegerType),
          context.tokens().GetIntegerLiteral(
              context.parse_tree().node_token(parse_node))});
  return true;
}

auto HandleFloatingPointLiteral(Context& context, Parse::Node parse_node)
    -> bool {
  context.AddInstAndPush(
      parse_node,
      SemIR::RealLiteral{
          parse_node,
          context.GetBuiltinType(SemIR::BuiltinKind::FloatingPointType),
          context.tokens().GetRealLiteral(
              context.parse_tree().node_token(parse_node))});
  return true;
}

auto HandleStringLiteral(Context& context, Parse::Node parse_node) -> bool {
  context.AddInstAndPush(
      parse_node,
      SemIR::StringLiteral{
          parse_node, context.GetBuiltinType(SemIR::BuiltinKind::StringType),
          context.tokens().GetStringLiteral(
              context.parse_tree().node_token(parse_node))});
  return true;
}

auto HandleBoolTypeLiteral(Context& context, Parse::Node parse_node) -> bool {
  context.node_stack().Push(parse_node, SemIR::InstId::BuiltinBoolType);
  return true;
}

auto HandleIntegerTypeLiteral(Context& context, Parse::Node parse_node)
    -> bool {
  auto text = context.tokens().GetTokenText(
      context.parse_tree().node_token(parse_node));
  if (text != "i32") {
    return context.TODO(parse_node, "Currently only i32 is allowed");
  }
  context.node_stack().Push(parse_node, SemIR::InstId::BuiltinIntegerType);
  return true;
}

auto HandleUnsignedIntegerTypeLiteral(Context& context, Parse::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "Need to support unsigned type literals");
}

auto HandleFloatingPointTypeLiteral(Context& context, Parse::Node parse_node)
    -> bool {
  auto text = context.tokens().GetTokenText(
      context.parse_tree().node_token(parse_node));
  if (text != "f64") {
    return context.TODO(parse_node, "Currently only f64 is allowed");
  }
  context.node_stack().Push(parse_node,
                            SemIR::InstId::BuiltinFloatingPointType);
  return true;
}

auto HandleStringTypeLiteral(Context& context, Parse::Node parse_node) -> bool {
  context.node_stack().Push(parse_node, SemIR::InstId::BuiltinStringType);
  return true;
}

auto HandleTypeTypeLiteral(Context& context, Parse::Node parse_node) -> bool {
  context.node_stack().Push(parse_node, SemIR::InstId::BuiltinTypeType);
  return true;
}

}  // namespace Carbon::Check
