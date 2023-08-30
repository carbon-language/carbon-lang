// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/function_context.h"

namespace Carbon::Lower {

auto HandleBindValue(FunctionContext& context, SemIR::NodeId node_id,
                     SemIR::Node node) -> void {
  switch (auto rep = SemIR::GetValueRepresentation(context.semantics_ir(),
                                                   node.type_id());
          rep.kind) {
    case SemIR::ValueRepresentation::None:
      // Nothing should use this value, but StubReference needs a value to
      // propagate.
      context.SetLocal(node_id,
                       llvm::PoisonValue::get(context.GetType(node.type_id())));
      break;
    case SemIR::ValueRepresentation::Copy:
      context.SetLocal(node_id, context.builder().CreateLoad(
                                    context.GetType(node.type_id()),
                                    context.GetLocal(node.GetAsBindValue())));
      break;
    case SemIR::ValueRepresentation::Pointer:
      context.SetLocal(node_id, context.GetLocal(node.GetAsBindValue()));
      break;
    case SemIR::ValueRepresentation::Custom:
      CARBON_FATAL() << "TODO: Add support for BindValue with custom value rep";
  }
}

auto HandleTemporary(FunctionContext& context, SemIR::NodeId node_id,
                     SemIR::Node node) -> void {
  auto [temporary_id, init_id] = node.GetAsTemporary();
  context.FinishInitialization(node.type_id(), temporary_id, init_id);
  context.SetLocal(node_id, context.GetLocal(temporary_id));
}

auto HandleTemporaryStorage(FunctionContext& context, SemIR::NodeId node_id,
                            SemIR::Node node) -> void {
  context.SetLocal(
      node_id, context.builder().CreateAlloca(context.GetType(node.type_id()),
                                              nullptr, "temp"));
}

}  // namespace Carbon::Lower
