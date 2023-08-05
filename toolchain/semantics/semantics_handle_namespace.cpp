// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

auto SemanticsHandleNamespaceStart(SemanticsContext& context,
                                   ParseTree::Node /*parse_node*/) -> bool {
  context.declaration_name_stack().Push();
  return true;
}

auto SemanticsHandleNamespace(SemanticsContext& context,
                              ParseTree::Node parse_node) -> bool {
  auto name_context = context.declaration_name_stack().Pop();
  auto namespace_id = context.AddNode(SemanticsNode::Namespace::Make(
      parse_node, context.semantics_ir().AddNameScope()));
  context.declaration_name_stack().AddNameToLookup(name_context, namespace_id);
  return true;
}

}  // namespace Carbon
