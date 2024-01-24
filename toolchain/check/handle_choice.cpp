// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleChoiceDefinition(Context& context,
                            Parse::ChoiceDefinitionId parse_node) -> bool {
  return context.TODO(parse_node, "HandleChoiceDefinition");
}

auto HandleChoiceIntroducer(Context& context,
                            Parse::ChoiceIntroducerId parse_node) -> bool {
  return context.TODO(parse_node, "HandleChoiceIntroducer");
}

auto HandleChoiceDefinitionStart(Context& context,
                                 Parse::ChoiceDefinitionStartId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleChoiceDefinitionStart");
}

auto HandleChoiceAlternativeListComma(
    Context& context, Parse::ChoiceAlternativeListCommaId parse_node) -> bool {
  return context.TODO(parse_node, "HandleChoiceAlternativeListComma");
}

}  // namespace Carbon::Check
