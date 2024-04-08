// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/call.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/typed_insts.h"

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

// Forms an IntLiteral instruction with type `i32` for a given literal integer
// value, which is assumed to be unsigned.
static auto MakeI32Literal(Context& context, Parse::NodeId node_id,
                           IntId int_id) -> SemIR::InstId {
  auto val = context.ints().Get(int_id);
  if (val.getActiveBits() > 31) {
    CARBON_DIAGNOSTIC(IntLiteralTooLargeForI32, Error,
                      "Integer literal with value {0} does not fit in i32.",
                      llvm::APSInt);
    context.emitter().Emit(node_id, IntLiteralTooLargeForI32,
                           llvm::APSInt(val, /*isUnsigned=*/true));
    return SemIR::InstId::BuiltinError;
  }
  // Literals are always represented as unsigned, so zero-extend if needed.
  auto i32_val = val.zextOrTrunc(32);
  return context.AddInst(
      {node_id,
       SemIR::IntLiteral{context.GetBuiltinType(SemIR::BuiltinKind::IntType),
                         context.ints().Add(i32_val)}});
}

auto HandleIntLiteral(Context& context, Parse::IntLiteralId node_id) -> bool {
  // Convert the literal to i32.
  // TODO: Form an integer literal value and a corresponding type here instead.
  auto int_literal_id = MakeI32Literal(
      context, node_id,
      context.tokens().GetIntLiteral(context.parse_tree().node_token(node_id)));
  context.node_stack().Push(node_id, int_literal_id);
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
  // TODO: Migrate once functions can be in prelude.carbon.
  // auto fn_inst_id = context.LookupNameInCore(node_id, "Bool");
  // auto type_inst_id = PerformCall(context, node_id, fn_inst_id, {});
  // context.node_stack().Push(node_id, type_inst_id);
  context.node_stack().Push(node_id, SemIR::InstId::BuiltinBoolType);
  return true;
}

// Shared implementation for handling `iN` and `uN` literals.
static auto HandleIntOrUnsignedIntTypeLiteral(Context& context,
                                              Parse::NodeId node_id,
                                              SemIR::IntKind int_kind,
                                              IntId size_id) -> bool {
  if (!(context.ints().Get(size_id) & 3).isZero()) {
    CARBON_DIAGNOSTIC(IntWidthNotMultipleOf8, Error,
                      "Bit width of integer type literal must be a multiple of "
                      "8. Use `Core.{0}({1})` instead.",
                      std::string, llvm::APSInt);
    context.emitter().Emit(
        node_id, IntWidthNotMultipleOf8, int_kind.is_signed() ? "Int" : "UInt",
        llvm::APSInt(context.ints().Get(size_id), /*isUnsigned=*/true));
  }
  // TODO: Migrate to a call to `Core.Int` or `Core.UInt`.
  auto width_id = MakeI32Literal(context, node_id, size_id);
  context.AddInstAndPush(
      {node_id, SemIR::IntType{.type_id = context.GetBuiltinType(
                                   SemIR::BuiltinKind::TypeType),
                               .int_kind = int_kind,
                               .bit_width_id = width_id}});
  return true;
}

auto HandleIntTypeLiteral(Context& context, Parse::IntTypeLiteralId node_id)
    -> bool {
  auto tok_id = context.parse_tree().node_token(node_id);
  auto size_id = context.tokens().GetTypeLiteralSize(tok_id);
  // Special case: `i32` has a custom builtin for now.
  // TODO: Remove this special case.
  if (context.ints().Get(size_id) == 32) {
    context.node_stack().Push(node_id, SemIR::InstId::BuiltinIntType);
    return true;
  }
  return HandleIntOrUnsignedIntTypeLiteral(context, node_id,
                                           SemIR::IntKind::Signed, size_id);
}

auto HandleUnsignedIntTypeLiteral(Context& context,
                                  Parse::UnsignedIntTypeLiteralId node_id)
    -> bool {
  auto tok_id = context.parse_tree().node_token(node_id);
  auto size_id = context.tokens().GetTypeLiteralSize(tok_id);
  return HandleIntOrUnsignedIntTypeLiteral(context, node_id,
                                           SemIR::IntKind::Unsigned, size_id);
}

auto HandleFloatTypeLiteral(Context& context, Parse::FloatTypeLiteralId node_id)
    -> bool {
  auto text =
      context.tokens().GetTokenText(context.parse_tree().node_token(node_id));
  if (text != "f64") {
    return context.TODO(node_id, "Currently only f64 is allowed");
  }
  // TODO: Migrate once functions can be in prelude.carbon.
  // auto fn_inst_id = context.LookupNameInCore(node_id, "Float");
  // auto width_inst_id = context.AddInstInNoBlock(
  //     {node_id,
  //      SemIR::IntLiteral{
  //          context.GetBuiltinType(SemIR::BuiltinKind::IntType),
  //          context.ints().Add(llvm::APInt(/*numBits=*/32, /*val=*/64))}});
  // auto type_inst_id =
  //     PerformCall(context, node_id, fn_inst_id, {width_inst_id});
  // context.node_stack().Push(node_id, type_inst_id);
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
