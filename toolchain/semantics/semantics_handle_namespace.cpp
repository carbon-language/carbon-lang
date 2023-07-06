// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

auto SemanticsHandleNamespaceStart(SemanticsContext& context,
                                   ParseTree::Node /*parse_node*/) -> bool {
  context.PushDeclarationName();
  return true;
}

auto SemanticsHandleNamespace(SemanticsContext& context,
                              ParseTree::Node parse_node) -> bool {
  auto name_context = context.PopDeclarationName();
  auto namespace_id = context.AddNode(SemanticsNode::Namespace::Make(
      parse_node, context.semantics_ir().AddNameScope()));
  context.AddNameToLookup(name_context, namespace_id);
  return true;
}

}  // namespace Carbon
