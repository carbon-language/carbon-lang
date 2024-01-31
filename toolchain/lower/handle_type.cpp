// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/function_context.h"

namespace Carbon::Lower {

auto HandleArrayType(FunctionContext& context, SemIR::InstId inst_id,
                     SemIR::ArrayType /*inst*/) -> void {
  context.SetLocal(inst_id, context.GetTypeAsValue());
}

auto HandleClassType(FunctionContext& context, SemIR::InstId inst_id,
                     SemIR::ClassType /*inst*/) -> void {
  context.SetLocal(inst_id, context.GetTypeAsValue());
}

auto HandleConstType(FunctionContext& context, SemIR::InstId inst_id,
                     SemIR::ConstType /*inst*/) -> void {
  context.SetLocal(inst_id, context.GetTypeAsValue());
}

auto HandleInterfaceType(FunctionContext& context, SemIR::InstId inst_id,
                         SemIR::InterfaceType /*inst*/) -> void {
  context.SetLocal(inst_id, context.GetTypeAsValue());
}

auto HandlePointerType(FunctionContext& context, SemIR::InstId inst_id,
                       SemIR::PointerType /*inst*/) -> void {
  context.SetLocal(inst_id, context.GetTypeAsValue());
}

auto HandleStructType(FunctionContext& context, SemIR::InstId inst_id,
                      SemIR::StructType /*inst*/) -> void {
  context.SetLocal(inst_id, context.GetTypeAsValue());
}

auto HandleTupleType(FunctionContext& context, SemIR::InstId inst_id,
                     SemIR::TupleType /*inst*/) -> void {
  context.SetLocal(inst_id, context.GetTypeAsValue());
}

auto HandleUnboundElementType(FunctionContext& context, SemIR::InstId inst_id,
                              SemIR::UnboundElementType /*inst*/) -> void {
  context.SetLocal(inst_id, context.GetTypeAsValue());
}

}  // namespace Carbon::Lower
