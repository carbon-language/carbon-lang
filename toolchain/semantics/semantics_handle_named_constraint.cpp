// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

auto SemanticsHandleNamedConstraintDeclaration(SemanticsContext& context,
                                               ParseTree::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleNamedConstraintDeclaration");
}

auto SemanticsHandleNamedConstraintDefinition(SemanticsContext& context,
                                              ParseTree::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleNamedConstraintDefinition");
}

auto SemanticsHandleNamedConstraintDefinitionStart(SemanticsContext& context,
                                                   ParseTree::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleNamedConstraintDefinitionStart");
}

auto SemanticsHandleNamedConstraintIntroducer(SemanticsContext& context,
                                              ParseTree::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleNamedConstraintIntroducer");
}

}  // namespace Carbon
