// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

auto SemanticsHandleVariableDeclaration(SemanticsContext& context,
                                        ParseTree::Node parse_node) -> bool {
  // Handle the optional initializer.
  auto expr_node_id = SemanticsNodeId::Invalid;
  bool has_init =
      context.parse_tree().node_kind(context.node_stack().PeekParseNode()) !=
      ParseNodeKind::PatternBinding;
  if (has_init) {
    expr_node_id = context.node_stack().Pop<SemanticsNodeId>();
    context.node_stack().PopAndDiscardSoloParseNode(
        ParseNodeKind::VariableInitializer);
  }

  // Get the storage and add it to name lookup.
  auto binding_id =
      context.node_stack().Pop<SemanticsNodeId>(ParseNodeKind::PatternBinding);
  auto binding = context.semantics_ir().GetNode(binding_id);
  auto [name_id, storage_id] = binding.GetAsBindName();
  context.AddNameToLookup(binding.parse_node(), name_id, storage_id);

  // If there was an initializer, assign it to storage.
  if (has_init) {
    auto cast_value_id = context.ImplicitAsRequired(
        parse_node, expr_node_id,
        context.semantics_ir().GetNode(storage_id).type_id());
    context.AddNode(SemanticsNode::Assign::Make(
        parse_node, context.semantics_ir().GetNode(cast_value_id).type_id(),
        storage_id, cast_value_id));
  }

  context.node_stack().PopAndDiscardSoloParseNode(
      ParseNodeKind::VariableIntroducer);

  return true;
}

auto SemanticsHandleVariableIntroducer(SemanticsContext& context,
                                       ParseTree::Node parse_node) -> bool {
  // No action, just a bracketing node.
  context.node_stack().Push(parse_node);
  return true;
}

auto SemanticsHandleVariableInitializer(SemanticsContext& context,
                                        ParseTree::Node parse_node) -> bool {
  // No action, just a bracketing node.
  context.node_stack().Push(parse_node);
  return true;
}

}  // namespace Carbon
