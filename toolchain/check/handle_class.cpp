// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleClassDeclaration(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleClassDeclaration");
}

auto HandleClassDefinition(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleClassDefinition");
}

auto HandleClassDefinitionStart(Context& context, Parse::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleClassDefinitionStart");
}

auto HandleClassIntroducer(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleClassIntroducer");
}

}  // namespace Carbon::Check
