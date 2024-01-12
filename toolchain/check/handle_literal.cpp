// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleBoolLiteralFalse(Context& context,
                            Parse::BoolLiteralFalseId parse_node) -> bool {
  context.AddInstAndPush(
      {parse_node,
       SemIR::BoolLiteral{context.GetBuiltinType(SemIR::BuiltinKind::BoolType),
                          SemIR::BoolValue::False}});
  return true;
}

auto HandleBoolLiteralTrue(Context& context,
                           Parse::BoolLiteralTrueId parse_node) -> bool {
  context.AddInstAndPush(
      {parse_node,
       SemIR::BoolLiteral{context.GetBuiltinType(SemIR::BuiltinKind::BoolType),
                          SemIR::BoolValue::True}});
  return true;
}

auto HandleIntLiteral(Context& context, Parse::IntLiteralId parse_node)
    -> bool {
  context.AddInstAndPush(
      {parse_node,
       SemIR::IntLiteral{context.GetBuiltinType(SemIR::BuiltinKind::IntType),
                         context.tokens().GetIntLiteral(
                             context.parse_tree().node_token(parse_node))}});
  return true;
}

auto HandleRealLiteral(Context& context, Parse::RealLiteralId parse_node)
    -> bool {
  context.AddInstAndPush(
      {parse_node,
       SemIR::RealLiteral{context.GetBuiltinType(SemIR::BuiltinKind::FloatType),
                          context.tokens().GetRealLiteral(
                              context.parse_tree().node_token(parse_node))}});
  return true;
}

auto HandleStringLiteral(Context& context, Parse::StringLiteralId parse_node)
    -> bool {
  context.AddInstAndPush(
      {parse_node, SemIR::StringLiteral{
                       context.GetBuiltinType(SemIR::BuiltinKind::StringType),
                       context.tokens().GetStringLiteralValue(
                           context.parse_tree().node_token(parse_node))}});
  return true;
}

auto HandleBoolTypeLiteral(Context& context,
                           Parse::BoolTypeLiteralId parse_node) -> bool {
  context.node_stack().Push(parse_node, SemIR::InstId::BuiltinBoolType);
  return true;
}

auto HandleIntTypeLiteral(Context& context, Parse::IntTypeLiteralId parse_node)
    -> bool {
  auto text = context.tokens().GetTokenText(
      context.parse_tree().node_token(parse_node));
  if (text != "i32") {
    return context.TODO(parse_node, "Currently only i32 is allowed");
  }
  context.node_stack().Push(parse_node, SemIR::InstId::BuiltinIntType);
  return true;
}

auto HandleUnsignedIntTypeLiteral(Context& context,
                                  Parse::UnsignedIntTypeLiteralId parse_node)
    -> bool {
  return context.TODO(parse_node, "Need to support unsigned type literals");
}

auto HandleFloatTypeLiteral(Context& context,
                            Parse::FloatTypeLiteralId parse_node) -> bool {
  auto text = context.tokens().GetTokenText(
      context.parse_tree().node_token(parse_node));
  if (text != "f64") {
    return context.TODO(parse_node, "Currently only f64 is allowed");
  }
  context.node_stack().Push(parse_node, SemIR::InstId::BuiltinFloatType);
  return true;
}

auto HandleStringTypeLiteral(Context& context,
                             Parse::StringTypeLiteralId parse_node) -> bool {
  context.node_stack().Push(parse_node, SemIR::InstId::BuiltinStringType);
  return true;
}

auto HandleTypeTypeLiteral(Context& context,
                           Parse::TypeTypeLiteralId parse_node) -> bool {
  context.node_stack().Push(parse_node, SemIR::InstId::BuiltinTypeType);
  return true;
}

}  // namespace Carbon::Check
