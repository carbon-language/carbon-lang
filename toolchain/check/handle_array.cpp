// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/parse/node_kind.h"

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
  auto [element_type_node_id, element_type_inst_id] =
      context.node_stack().PopExprWithParseNode();

  // The array bound must be a constant.
  //
  // TODO: Should we support runtime-phase bounds in cases such as:
  //   comptime fn F(n: i32) -> type { return [i32; n]; }
  auto bound_inst = context.constant_values().Get(bound_inst_id);
  if (!bound_inst.is_constant()) {
    CARBON_DIAGNOSTIC(InvalidArrayExpr, Error,
                      "Array bound is not a constant.");
    context.emitter().Emit(bound_inst_id, InvalidArrayExpr);
    context.node_stack().Push(parse_node, SemIR::InstId::BuiltinError);
    return true;
  }

  context.AddInstAndPush(
      {parse_node, SemIR::ArrayType{SemIR::TypeId::TypeType, bound_inst_id,
                                    ExprAsType(context, element_type_node_id,
                                               element_type_inst_id)}});
  return true;
}

}  // namespace Carbon::Check
