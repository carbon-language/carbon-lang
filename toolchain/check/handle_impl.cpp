// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleImplIntroducer(Context& context, Parse::ImplIntroducerId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleImplIntroducer");
}

auto HandleImplForall(Context& context, Parse::ImplForallId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleImplForall");
}

auto HandleTypeImplAs(Context& context, Parse::TypeImplAsId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleTypeImplAs");
}

auto HandleDefaultSelfImplAs(Context& context,
                             Parse::DefaultSelfImplAsId parse_node) -> bool {
  return context.TODO(parse_node, "HandleDefaultSelfImplAs");
}

auto HandleImplDecl(Context& context, Parse::ImplDeclId parse_node) -> bool {
  return context.TODO(parse_node, "HandleImplDecl");
}

auto HandleImplDefinitionStart(Context& context,
                               Parse::ImplDefinitionStartId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleImplDefinitionStart");
}

auto HandleImplDefinition(Context& context, Parse::ImplDefinitionId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleImplDefinition");
}

}  // namespace Carbon::Check
