// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"

namespace Carbon {

auto SemanticsHandleClassDeclaration(SemanticsContext& context,
                                     ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleClassDeclaration");
}

auto SemanticsHandleClassDefinition(SemanticsContext& context,
                                    ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleClassDefinition");
}

auto SemanticsHandleClassDefinitionStart(SemanticsContext& context,
                                         ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleClassDefinitionStart");
}

auto SemanticsHandleClassIntroducer(SemanticsContext& context,
                                    ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleClassIntroducer");
}

}  // namespace Carbon
