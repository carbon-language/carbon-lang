// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lowering/lowering_function_context.h"

namespace Carbon {

auto LoweringHandleInitializeFrom(LoweringFunctionContext& context,
                                  SemanticsNodeId node_id, SemanticsNode node)
    -> void {
  // TODO: If the value representation is indirect, perform the copy.
  auto [init_value_id, target_id] = node.GetAsInitializeFrom();
  context.SetLocal(node_id, context.GetLocal(init_value_id));
}

auto LoweringHandleMaterializeTemporary(LoweringFunctionContext& context,
                                        SemanticsNodeId node_id,
                                        SemanticsNode node) -> void {
  context.SetLocal(node_id, context.GetLocal(node.GetAsMaterializeTemporary()));
}

auto LoweringHandleValueBinding(LoweringFunctionContext& context,
                                SemanticsNodeId node_id, SemanticsNode node)
    -> void {
  context.SetLocal(node_id, context.GetLocalLoaded(node.GetAsValueBinding()));
}

}  // namespace Carbon
