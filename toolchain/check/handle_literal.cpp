// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleBoolLiteralFalse(Context& context, Parse::BoolLiteralFalseId node_id)
    -> bool {
  context.AddInstAndPush(
      {node_id,
       SemIR::BoolLiteral{context.GetBuiltinType(SemIR::BuiltinKind::BoolType),
                          SemIR::BoolValue::False}});
  return true;
}

auto HandleBoolLiteralTrue(Context& context, Parse::BoolLiteralTrueId node_id)
    -> bool {
  context.AddInstAndPush(
      {node_id,
       SemIR::BoolLiteral{context.GetBuiltinType(SemIR::BuiltinKind::BoolType),
                          SemIR::BoolValue::True}});
  return true;
}

auto HandleIntLiteral(Context& context, Parse::IntLiteralId node_id) -> bool {
  // Convert the literal to i32.
  // TODO: Form an integer literal value and a corresponding type here instead.
  auto literal_int_id =
      context.tokens().GetIntLiteral(context.parse_tree().node_token(node_id));
  auto literal_val = context.ints().Get(literal_int_id);
  if (literal_val.getActiveBits() > 31) {
    CARBON_DIAGNOSTIC(IntLiteralTooLargeForI32, Error,
                      "Integer literal with value {0} does not fit in i32.",
                      llvm::APSInt);
    context.emitter().Emit(node_id, IntLiteralTooLargeForI32,
                           llvm::APSInt(literal_val, /*isUnsigned=*/true));
    context.node_stack().Push(node_id, SemIR::InstId::BuiltinError);
    return true;
  }
  // Literals are always represented as unsigned, so zero-extend if needed.
  auto i32_val = literal_val.zextOrTrunc(32);
  context.AddInstAndPush(
      {node_id,
       SemIR::IntLiteral{context.GetBuiltinType(SemIR::BuiltinKind::IntType),
                         context.ints().Add(i32_val)}});
  return true;
}

auto HandleRealLiteral(Context& context, Parse::RealLiteralId node_id) -> bool {
  context.AddInstAndPush(
      {node_id,
       SemIR::RealLiteral{context.GetBuiltinType(SemIR::BuiltinKind::FloatType),
                          context.tokens().GetRealLiteral(
                              context.parse_tree().node_token(node_id))}});
  return true;
}

auto HandleStringLiteral(Context& context, Parse::StringLiteralId node_id)
    -> bool {
  context.AddInstAndPush(
      {node_id, SemIR::StringLiteral{
                    context.GetBuiltinType(SemIR::BuiltinKind::StringType),
                    context.tokens().GetStringLiteralValue(
                        context.parse_tree().node_token(node_id))}});
  return true;
}

auto HandleBoolTypeLiteral(Context& context, Parse::BoolTypeLiteralId node_id)
    -> bool {
  context.node_stack().Push(node_id, SemIR::InstId::BuiltinBoolType);
  return true;
}

auto HandleIntTypeLiteral(Context& context, Parse::IntTypeLiteralId node_id)
    -> bool {
  auto text =
      context.tokens().GetTokenText(context.parse_tree().node_token(node_id));
  if (text != "i32") {
    return context.TODO(node_id, "Currently only i32 is allowed");
  }
  context.node_stack().Push(node_id, SemIR::InstId::BuiltinIntType);
  return true;
}

auto HandleUnsignedIntTypeLiteral(Context& context,
                                  Parse::UnsignedIntTypeLiteralId node_id)
    -> bool {
  return context.TODO(node_id, "Need to support unsigned type literals");
}

auto HandleFloatTypeLiteral(Context& context, Parse::FloatTypeLiteralId node_id)
    -> bool {
  auto text =
      context.tokens().GetTokenText(context.parse_tree().node_token(node_id));
  if (text != "f64") {
    return context.TODO(node_id, "Currently only f64 is allowed");
  }
  context.node_stack().Push(node_id, SemIR::InstId::BuiltinFloatType);
  return true;
}

auto HandleStringTypeLiteral(Context& context,
                             Parse::StringTypeLiteralId node_id) -> bool {
  context.node_stack().Push(node_id, SemIR::InstId::BuiltinStringType);
  return true;
}

auto HandleTypeTypeLiteral(Context& context, Parse::TypeTypeLiteralId node_id)
    -> bool {
  context.node_stack().Push(node_id, SemIR::InstId::BuiltinTypeType);
  return true;
}

auto HandleAutoTypeLiteral(Context& context, Parse::AutoTypeLiteralId node_id)
    -> bool {
  return context.TODO(node_id, "HandleAutoTypeLiteral");
}

}  // namespace Carbon::Check
