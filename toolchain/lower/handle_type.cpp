// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/function_context.h"

namespace Carbon::Lower {

auto HandleArrayType(FunctionContext& context, SemIR::NodeId node_id,
                     SemIR::ArrayType /*node*/) -> void {
  context.SetLocal(node_id, context.GetTypeAsValue());
}

auto HandleConstType(FunctionContext& context, SemIR::NodeId node_id,
                     SemIR::ConstType /*node*/) -> void {
  context.SetLocal(node_id, context.GetTypeAsValue());
}

auto HandlePointerType(FunctionContext& context, SemIR::NodeId node_id,
                       SemIR::PointerType /*node*/) -> void {
  context.SetLocal(node_id, context.GetTypeAsValue());
}

auto HandleStructType(FunctionContext& context, SemIR::NodeId node_id,
                      SemIR::StructType /*node*/) -> void {
  context.SetLocal(node_id, context.GetTypeAsValue());
}

auto HandleTupleType(FunctionContext& context, SemIR::NodeId node_id,
                     SemIR::TupleType /*node*/) -> void {
  context.SetLocal(node_id, context.GetTypeAsValue());
}

}  // namespace Carbon::Lower
