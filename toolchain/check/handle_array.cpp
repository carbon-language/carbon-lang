// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"
#include "toolchain/parse/node_kind.h"

namespace Carbon::Check {

auto HandleParseNode(Context& /*context*/, Parse::ArrayExprStartId /*node_id*/)
    -> bool {
  return true;
}

auto HandleParseNode(Context& context, Parse::ArrayExprSemiId node_id) -> bool {
  context.node_stack().Push(node_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::ArrayExprId node_id) -> bool {
  // TODO: Handle array type with undefined bound.
  if (context.node_stack()
          .PopAndDiscardSoloNodeIdIf<Parse::NodeKind::ArrayExprSemi>()) {
    context.node_stack().PopAndIgnore();
    return context.TODO(node_id, "HandleArrayExprWithoutBounds");
  }

  auto bound_inst_id = context.node_stack().PopExpr();
  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::ArrayExprSemi>();
  auto [element_type_node_id, element_type_inst_id] =
      context.node_stack().PopExprWithNodeId();

  // The array bound must be a constant.
  //
  // TODO: Should we support runtime-phase bounds in cases such as:
  //   comptime fn F(n: i32) -> type { return [i32; n]; }
  auto bound_inst = context.constant_values().Get(bound_inst_id);
  if (!bound_inst.is_constant()) {
    CARBON_DIAGNOSTIC(InvalidArrayExpr, Error, "array bound is not a constant");
    context.emitter().Emit(bound_inst_id, InvalidArrayExpr);
    context.node_stack().Push(node_id, SemIR::InstId::BuiltinError);
    return true;
  }

  context.AddInstAndPush<SemIR::ArrayType>(
      node_id, {.type_id = SemIR::TypeId::TypeType,
                .bound_id = bound_inst_id,
                .element_type_id = ExprAsType(context, element_type_node_id,
                                              element_type_inst_id)
                                       .type_id});
  return true;
}

}  // namespace Carbon::Check
