// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"

namespace Carbon::Check {

auto HandleArrayExpressionStart(Context& /*context*/,
                                Parse::Lamp /*parse_lamp*/) -> bool {
  return true;
}

auto HandleArrayExpressionSemi(Context& context, Parse::Lamp parse_lamp)
    -> bool {
  context.lamp_stack().Push(parse_lamp);
  return true;
}

auto HandleArrayExpression(Context& context, Parse::Lamp parse_lamp) -> bool {
  // TODO: Handle array type with undefined bound.
  if (context.parse_tree().node_kind(context.lamp_stack().PeekParseLamp()) ==
      Parse::LampKind::ArrayExpressionSemi) {
    context.lamp_stack().PopAndIgnore();
    context.lamp_stack().PopAndIgnore();
    return context.TODO(parse_lamp, "HandleArrayExpressionWithoutBounds");
  }

  SemIR::InstId bound_inst_id = context.lamp_stack().PopExpression();
  context.lamp_stack()
      .PopAndDiscardSoloParseLamp<Parse::LampKind::ArrayExpressionSemi>();
  SemIR::InstId element_type_inst_id = context.lamp_stack().PopExpression();
  auto bound_inst = context.insts().Get(bound_inst_id);
  if (auto literal = bound_inst.TryAs<SemIR::IntegerLiteral>()) {
    const auto& bound_value = context.integers().Get(literal->integer_id);
    // TODO: Produce an error if the array type is too large.
    if (bound_value.getActiveBits() <= 64) {
      context.AddInstAndPush(
          parse_lamp,
          SemIR::ArrayType{
              parse_lamp, SemIR::TypeId::TypeType, bound_inst_id,
              ExpressionAsType(context, parse_lamp, element_type_inst_id)});
      return true;
    }
  }
  CARBON_DIAGNOSTIC(InvalidArrayExpression, Error, "Invalid array expression.");
  context.emitter().Emit(parse_lamp, InvalidArrayExpression);
  context.lamp_stack().Push(parse_lamp, SemIR::InstId::BuiltinError);
  return true;
}

}  // namespace Carbon::Check
