// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleArrayExprStart(Context& /*context*/,
                          Parse::ArrayExprStartId /*parse_node*/) -> bool {
  return true;
}

auto HandleArrayExprSemi(Context& context, Parse::ArrayExprSemiId parse_node)
    -> bool {
  context.node_stack().Push(parse_node);
  return true;
}

auto HandleArrayExpr(Context& context, Parse::ArrayExprId parse_node) -> bool {
  // TODO: Handle array type with undefined bound.
  if (context.node_stack()
          .PopAndDiscardSoloParseNodeIf<Parse::NodeKind::ArrayExprSemi>()) {
    context.node_stack().PopAndIgnore();
    return context.TODO(parse_node, "HandleArrayExprWithoutBounds");
  }

  auto bound_inst_id = context.node_stack().PopExpr();
  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::ArrayExprSemi>();
  auto element_type_inst_id = context.node_stack().PopExpr();
  auto bound_inst = context.insts().Get(bound_inst_id);
  if (auto literal = bound_inst.TryAs<SemIR::IntLiteral>()) {
    const auto& bound_value = context.ints().Get(literal->int_id);
    // TODO: Produce an error if the array type is too large.
    if (bound_value.getActiveBits() <= 64) {
      context.AddInstAndPush(
          parse_node, SemIR::ArrayType{SemIR::TypeId::TypeType, bound_inst_id,
                                       ExprAsType(context, parse_node,
                                                  element_type_inst_id)});
      return true;
    }
  }
  CARBON_DIAGNOSTIC(InvalidArrayExpr, Error, "Invalid array expression.");
  context.emitter().Emit(parse_node, InvalidArrayExpr);
  context.node_stack().Push(parse_node, SemIR::InstId::BuiltinError);
  return true;
}

}  // namespace Carbon::Check
