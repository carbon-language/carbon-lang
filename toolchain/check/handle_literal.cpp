// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/call.h"
#include "toolchain/check/context.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/literal.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/type.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::BoolLiteralFalseId node_id)
    -> bool {
  AddInstAndPush<SemIR::BoolLiteral>(
      context, node_id,
      {.type_id = GetSingletonType(context, SemIR::BoolType::SingletonInstId),
       .value = SemIR::BoolValue::False});
  return true;
}

auto HandleParseNode(Context& context, Parse::BoolLiteralTrueId node_id)
    -> bool {
  AddInstAndPush<SemIR::BoolLiteral>(
      context, node_id,
      {.type_id = GetSingletonType(context, SemIR::BoolType::SingletonInstId),
       .value = SemIR::BoolValue::True});
  return true;
}

auto HandleParseNode(Context& context, Parse::IntLiteralId node_id) -> bool {
  auto int_literal_id = MakeIntLiteral(
      context, node_id,
      context.tokens().GetIntLiteral(context.parse_tree().node_token(node_id)));
  context.node_stack().Push(node_id, int_literal_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::RealLiteralId node_id) -> bool {
  // Convert the real literal to an llvm::APFloat and add it to the floats
  // ValueStore. In the future this would use an arbitrary precision Rational
  // type.
  //
  // TODO: Implement Carbon's actual implicit conversion rules for
  // floating-point constants, as per the design
  // docs/design/expressions/implicit_conversions.md
  auto real_id =
      context.tokens().GetRealLiteral(context.parse_tree().node_token(node_id));
  auto real_value = context.sem_ir().reals().Get(real_id);

  if (real_value.mantissa.getActiveBits() > 64) {
    CARBON_DIAGNOSTIC(RealMantissaTooLargeForI64, Error,
                      "real mantissa with value {0} does not fit in i64",
                      llvm::APSInt);
    context.emitter().Emit(node_id, RealMantissaTooLargeForI64,
                           llvm::APSInt(real_value.mantissa, true));
    context.node_stack().Push(node_id, SemIR::ErrorInst::SingletonInstId);
    return true;
  }

  if (real_value.exponent.getSignificantBits() > 64) {
    CARBON_DIAGNOSTIC(RealExponentTooLargeForI64, Error,
                      "real exponent with value {0} does not fit in i64",
                      llvm::APSInt);
    context.emitter().Emit(node_id, RealExponentTooLargeForI64,
                           llvm::APSInt(real_value.exponent, false));
    context.node_stack().Push(node_id, SemIR::ErrorInst::SingletonInstId);
    return true;
  }

  double double_val = real_value.mantissa.getZExtValue() *
                      std::pow((real_value.is_decimal ? 10 : 2),
                               real_value.exponent.getSExtValue());

  auto float_id = context.sem_ir().floats().Add(llvm::APFloat(double_val));
  AddInstAndPush<SemIR::FloatLiteral>(
      context, node_id,
      {.type_id =
           GetSingletonType(context, SemIR::LegacyFloatType::SingletonInstId),
       .float_id = float_id});
  return true;
}

auto HandleParseNode(Context& context, Parse::StringLiteralId node_id) -> bool {
  AddInstAndPush<SemIR::StringLiteral>(
      context, node_id,
      {.type_id = GetSingletonType(context, SemIR::StringType::SingletonInstId),
       .string_literal_id = context.tokens().GetStringLiteralValue(
           context.parse_tree().node_token(node_id))});
  return true;
}

auto HandleParseNode(Context& context, Parse::BoolTypeLiteralId node_id)
    -> bool {
  auto fn_inst_id = LookupNameInCore(context, node_id, "Bool");
  auto type_inst_id = PerformCall(context, node_id, fn_inst_id, {});
  context.node_stack().Push(node_id, type_inst_id);
  return true;
}

// Shared implementation for handling `iN` and `uN` literals.
static auto HandleIntOrUnsignedIntTypeLiteral(Context& context,
                                              Parse::NodeId node_id,
                                              SemIR::IntKind int_kind,
                                              IntId size_id) -> bool {
  if (!(context.ints().Get(size_id) & 3).isZero()) {
    CARBON_DIAGNOSTIC(IntWidthNotMultipleOf8, Error,
                      "bit width of integer type literal must be a multiple of "
                      "8; use `Core.{0:Int|UInt}({1})` instead",
                      Diagnostics::BoolAsSelect, llvm::APSInt);
    context.emitter().Emit(
        node_id, IntWidthNotMultipleOf8, int_kind.is_signed(),
        llvm::APSInt(context.ints().Get(size_id), /*isUnsigned=*/true));
  }
  auto type_inst_id = MakeIntTypeLiteral(context, node_id, int_kind, size_id);
  context.node_stack().Push(node_id, type_inst_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::IntTypeLiteralId node_id)
    -> bool {
  auto tok_id = context.parse_tree().node_token(node_id);
  auto size_id = context.tokens().GetTypeLiteralSize(tok_id);
  return HandleIntOrUnsignedIntTypeLiteral(context, node_id,
                                           SemIR::IntKind::Signed, size_id);
}

auto HandleParseNode(Context& context, Parse::UnsignedIntTypeLiteralId node_id)
    -> bool {
  auto tok_id = context.parse_tree().node_token(node_id);
  auto size_id = context.tokens().GetTypeLiteralSize(tok_id);
  return HandleIntOrUnsignedIntTypeLiteral(context, node_id,
                                           SemIR::IntKind::Unsigned, size_id);
}

auto HandleParseNode(Context& context, Parse::FloatTypeLiteralId node_id)
    -> bool {
  auto text =
      context.tokens().GetTokenText(context.parse_tree().node_token(node_id));
  if (text != "f64") {
    return context.TODO(node_id, "Currently only f64 is allowed");
  }
  auto tok_id = context.parse_tree().node_token(node_id);
  auto size_id = context.tokens().GetTypeLiteralSize(tok_id);
  auto width_id = MakeIntLiteral(context, node_id, size_id);
  auto fn_inst_id = LookupNameInCore(context, node_id, "Float");
  auto type_inst_id = PerformCall(context, node_id, fn_inst_id, {width_id});
  context.node_stack().Push(node_id, type_inst_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::StringTypeLiteralId node_id)
    -> bool {
  context.node_stack().Push(node_id, SemIR::StringType::SingletonInstId);
  return true;
}

auto HandleParseNode(Context& context, Parse::TypeTypeLiteralId node_id)
    -> bool {
  context.node_stack().Push(node_id, SemIR::TypeType::SingletonInstId);
  return true;
}

auto HandleParseNode(Context& context, Parse::AutoTypeLiteralId node_id)
    -> bool {
  return context.TODO(node_id, "HandleAutoTypeLiteral");
}

}  // namespace Carbon::Check
