// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleNamedConstraintDecl(Context& context,
                               Parse::NamedConstraintDeclId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleNamedConstraintDecl");
}

auto HandleNamedConstraintDefinition(
    Context& context, Parse::NamedConstraintDefinitionId parse_node) -> bool {
  return context.TODO(parse_node, "HandleNamedConstraintDefinition");
}

auto HandleNamedConstraintDefinitionStart(
    Context& context, Parse::NamedConstraintDefinitionStartId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleNamedConstraintDefinitionStart");
}

auto HandleNamedConstraintIntroducer(
    Context& context, Parse::NamedConstraintIntroducerId parse_node) -> bool {
  return context.TODO(parse_node, "HandleNamedConstraintIntroducer");
}

}  // namespace Carbon::Check
