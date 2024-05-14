// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleChoiceDefinition(Context& context, Parse::ChoiceDefinitionId node_id)
    -> bool {
  return context.TODO(node_id, "HandleChoiceDefinition");
}

auto HandleChoiceIntroducer(Context& context, Parse::ChoiceIntroducerId node_id)
    -> bool {
  return context.TODO(node_id, "HandleChoiceIntroducer");
}

auto HandleChoiceDefinitionStart(Context& context,
                                 Parse::ChoiceDefinitionStartId node_id)
    -> bool {
  return context.TODO(node_id, "HandleChoiceDefinitionStart");
}

auto HandleChoiceAlternativeListComma(
    Context& context, Parse::ChoiceAlternativeListCommaId node_id) -> bool {
  return context.TODO(node_id, "HandleChoiceAlternativeListComma");
}

}  // namespace Carbon::Check
