// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleNamedConstraintDeclaration(Context& context, Parse::Lamp parse_lamp)
    -> bool {
  return context.TODO(parse_lamp, "HandleNamedConstraintDeclaration");
}

auto HandleNamedConstraintDefinition(Context& context, Parse::Lamp parse_lamp)
    -> bool {
  return context.TODO(parse_lamp, "HandleNamedConstraintDefinition");
}

auto HandleNamedConstraintDefinitionStart(Context& context,
                                          Parse::Lamp parse_lamp) -> bool {
  return context.TODO(parse_lamp, "HandleNamedConstraintDefinitionStart");
}

auto HandleNamedConstraintIntroducer(Context& context, Parse::Lamp parse_lamp)
    -> bool {
  return context.TODO(parse_lamp, "HandleNamedConstraintIntroducer");
}

}  // namespace Carbon::Check
