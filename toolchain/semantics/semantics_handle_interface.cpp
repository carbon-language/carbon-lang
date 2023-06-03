// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"

namespace Carbon {

auto SemanticsHandleInterfaceDeclaration(SemanticsContext& context,
                                         ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleInterfaceDeclaration");
}

auto SemanticsHandleInterfaceDefinition(SemanticsContext& context,
                                        ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleInterfaceDefinition");
}

auto SemanticsHandleInterfaceDefinitionStart(SemanticsContext& context,
                                             ParseTree::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleInterfaceDefinitionStart");
}

auto SemanticsHandleInterfaceIntroducer(SemanticsContext& context,
                                        ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleInterfaceIntroducer");
}

}  // namespace Carbon
