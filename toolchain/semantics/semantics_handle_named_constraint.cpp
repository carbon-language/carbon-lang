// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon::Check {

auto HandleNamedConstraintDeclaration(Context& context,
                                      ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleNamedConstraintDeclaration");
}

auto HandleNamedConstraintDefinition(Context& context,
                                     ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleNamedConstraintDefinition");
}

auto HandleNamedConstraintDefinitionStart(Context& context,
                                          ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleNamedConstraintDefinitionStart");
}

auto HandleNamedConstraintIntroducer(Context& context,
                                     ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleNamedConstraintIntroducer");
}

}  // namespace Carbon::Check
