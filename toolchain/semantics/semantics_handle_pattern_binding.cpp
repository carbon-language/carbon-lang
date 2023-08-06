// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

auto SemanticsHandleAddress(SemanticsContext& context,
                            ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleAddress");
}

auto SemanticsHandleGenericPatternBinding(SemanticsContext& context,
                                          ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "GenericPatternBinding");
}

auto SemanticsHandlePatternBinding(SemanticsContext& context,
                                   ParseTree::Node parse_node) -> bool {
  auto [type_node, parsed_type_id] =
      context.node_stack().PopExpressionWithParseNode();
  auto cast_type_id = context.ExpressionAsType(type_node, parsed_type_id);

  // Get the name.
  auto [name_node, name_id] =
      context.node_stack().PopWithParseNode<ParseNodeKind::Name>();

  // Allocate storage, linked to the name for error locations.
  auto storage_id =
      context.AddNode(SemanticsNode::VarStorage::Make(name_node, cast_type_id));

  // Bind the name to storage.
  context.AddNodeAndPush(parse_node,
                         SemanticsNode::BindName::Make(name_node, cast_type_id,
                                                       name_id, storage_id));
  return true;
}

auto SemanticsHandleTemplate(SemanticsContext& context,
                             ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleTemplate");
}

}  // namespace Carbon
