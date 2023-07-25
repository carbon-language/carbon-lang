// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lowering/lowering_function_context.h"

namespace Carbon {

auto LoweringHandleStructType(LoweringFunctionContext& context,
                              SemanticsNodeId node_id, SemanticsNode /*node*/)
    -> void {
  context.SetLocal(node_id, context.GetTypeValue());
}

auto LoweringHandleTupleType(LoweringFunctionContext& context,
                             SemanticsNodeId node_id, SemanticsNode /*node*/)
    -> void {
  context.SetLocal(node_id, context.GetTypeValue());
}

}  // namespace Carbon
