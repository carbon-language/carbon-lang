// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleCodeBlockStart(Context& context, Parse::Lamp parse_lamp) -> bool {
  context.lamp_stack().Push(parse_lamp);
  context.PushScope();
  return true;
}

auto HandleCodeBlock(Context& context, Parse::Lamp /*parse_lamp*/) -> bool {
  context.PopScope();
  context.lamp_stack().PopForSoloParseLamp<Parse::LampKind::CodeBlockStart>();
  return true;
}

}  // namespace Carbon::Check
