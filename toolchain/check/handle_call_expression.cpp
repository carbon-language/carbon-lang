// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/node.h"

namespace Carbon::Check {

auto HandleCallExpression(Context& context, Parse::Node parse_node) -> bool {
  // Process the final explicit call argument, but leave the arguments block on
  // the stack until we add the return slot argument.
  context.ParamOrArgEndNoPop(Parse::NodeKind::CallExpressionStart);

  // TODO: Convert to call expression.
  auto [call_expr_parse_node, name_id] =
      context.node_stack()
          .PopWithParseNode<Parse::NodeKind::CallExpressionStart>();
  auto name_node = context.semantics_ir().GetNode(name_id);
  if (name_node.kind() != SemIR::NodeKind::FunctionDeclaration) {
    // TODO: Work on error.
    context.TODO(parse_node, "Not a callable name");
    context.node_stack().Push(parse_node, name_id);
    context.ParamOrArgPop();
    return true;
  }

  auto function_id = name_node.GetAsFunctionDeclaration();
  const auto& callable = context.semantics_ir().GetFunction(function_id);

  // For functions with an implicit return type, the return type is the empty
  // tuple type.
  SemIR::TypeId type_id = callable.return_type_id;
  if (!type_id.is_valid()) {
    type_id = context.CanonicalizeTupleType(call_expr_parse_node, {});
  }

  // If there is a return slot, add a corresponding argument.
  if (callable.return_slot_id.is_valid()) {
    // Tentatively put storage for a temporary in the function's return slot.
    // This will be replaced if necessary when we perform initialization.
    auto temp_id = context.AddNode(SemIR::Node::TemporaryStorage::Make(
        call_expr_parse_node, callable.return_type_id));
    context.ParamOrArgSave(temp_id);
  }

  // Convert the arguments to match the parameters.
  auto refs_id = context.ParamOrArgPop();
  if (!context.ImplicitAsForArgs(call_expr_parse_node, refs_id,
                                 name_node.parse_node(), callable.param_refs_id,
                                 callable.return_slot_id.is_valid())) {
    context.node_stack().Push(parse_node, SemIR::NodeId::BuiltinError);
    return true;
  }

  auto call_node_id = context.AddNode(SemIR::Node::Call::Make(
      call_expr_parse_node, type_id, refs_id, function_id));

  context.node_stack().Push(parse_node, call_node_id);
  return true;
}

auto HandleCallExpressionComma(Context& context, Parse::Node /*parse_node*/)
    -> bool {
  context.ParamOrArgComma();
  return true;
}

auto HandleCallExpressionStart(Context& context, Parse::Node parse_node)
    -> bool {
  auto name_id = context.node_stack().PopExpression();
  context.node_stack().Push(parse_node, name_id);
  context.ParamOrArgStart();
  return true;
}

}  // namespace Carbon::Check
