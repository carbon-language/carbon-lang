// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleInterfaceDeclaration(Context& context, Parse::Lamp parse_lamp)
    -> bool {
  return context.TODO(parse_lamp, "HandleInterfaceDeclaration");
}

auto HandleInterfaceDefinition(Context& context, Parse::Lamp parse_lamp)
    -> bool {
  return context.TODO(parse_lamp, "HandleInterfaceDefinition");
}

auto HandleInterfaceDefinitionStart(Context& context, Parse::Lamp parse_lamp)
    -> bool {
  return context.TODO(parse_lamp, "HandleInterfaceDefinitionStart");
}

auto HandleInterfaceIntroducer(Context& context, Parse::Lamp parse_lamp)
    -> bool {
  return context.TODO(parse_lamp, "HandleInterfaceIntroducer");
}

}  // namespace Carbon::Check
