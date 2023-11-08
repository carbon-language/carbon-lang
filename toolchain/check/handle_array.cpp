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
                                Parse::Node /*parse_node*/) -> bool {
  return true;
}

auto HandleArrayExpressionSemi(Context& context, Parse::Node parse_node)
    -> bool {
  context.node_stack().Push(parse_node);
  return true;
}

auto HandleArrayExpression(Context& context, Parse::Node parse_node) -> bool {
  // TODO: Handle array type with undefined bound.
  if (context.parse_tree().node_kind(context.node_stack().PeekParseNode()) ==
      Parse::NodeKind::ArrayExpressionSemi) {
    context.node_stack().PopAndIgnore();
    context.node_stack().PopAndIgnore();
    return context.TODO(parse_node, "HandleArrayExpressionWithoutBounds");
  }

  auto bound_inst_id = context.node_stack().PopExpression();
  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::ArrayExpressionSemi>();
  auto element_type_inst_id = context.node_stack().PopExpression();
  auto bound_inst = context.insts().Get(bound_inst_id);
  if (auto literal = bound_inst.TryAs<SemIR::IntegerLiteral>()) {
    const auto& bound_value = context.integers().Get(literal->integer_id);
    // TODO: Produce an error if the array type is too large.
    if (bound_value.getActiveBits() <= 64) {
      context.AddInstAndPush(
          parse_node,
          SemIR::ArrayType{
              parse_node, SemIR::TypeId::TypeType, bound_inst_id,
              ExpressionAsType(context, parse_node, element_type_inst_id)});
      return true;
    }
  }
  CARBON_DIAGNOSTIC(InvalidArrayExpression, Error, "Invalid array expression.");
  context.emitter().Emit(parse_node, InvalidArrayExpression);
  context.node_stack().Push(parse_node, SemIR::InstId::BuiltinError);
  return true;
}

}  // namespace Carbon::Check
