// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon::Check {

auto HandleVariableDeclaration(Context& context, Parse::Node parse_node)
    -> bool {
  // Handle the optional initializer.
  auto expr_node_id = SemIR::NodeId::Invalid;
  bool has_init =
      context.parse_tree().node_kind(context.node_stack().PeekParseNode()) !=
      Parse::NodeKind::PatternBinding;
  if (has_init) {
    expr_node_id = context.node_stack().PopExpression();
    context.node_stack()
        .PopAndDiscardSoloParseNode<Parse::NodeKind::VariableInitializer>();
  }

  // Get the storage and add it to name lookup.
  SemIR::NodeId var_id =
      context.node_stack().Pop<Parse::NodeKind::PatternBinding>();
  auto var = context.semantics_ir().GetNode(var_id);
  auto name_id = var.GetAsVarStorage();
  context.AddNameToLookup(var.parse_node(), name_id, var_id);
  // If there was an initializer, assign it to storage.
  if (has_init) {
    context.Initialize(parse_node, var_id, expr_node_id);
  }

  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::VariableIntroducer>();

  return true;
}

auto HandleVariableIntroducer(Context& context, Parse::Node parse_node)
    -> bool {
  // No action, just a bracketing node.
  context.node_stack().Push(parse_node);
  return true;
}

auto HandleVariableInitializer(Context& context, Parse::Node parse_node)
    -> bool {
  // No action, just a bracketing node.
  context.node_stack().Push(parse_node);
  return true;
}

}  // namespace Carbon::Check
