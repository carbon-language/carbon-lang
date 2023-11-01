// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleEmptyDeclaration(Context& /*context*/, Parse::Lamp /*parse_lamp*/)
    -> bool {
  // Empty declarations have no actions associated.
  return true;
}

auto HandleInvalidParse(Context& context, Parse::Lamp parse_lamp) -> bool {
  return context.TODO(parse_lamp, "HandleInvalidParse");
}

}  // namespace Carbon::Check
