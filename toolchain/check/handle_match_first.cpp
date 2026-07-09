// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::MatchFirstIntroducerId node_id)
    -> bool {
  return context.TODO(node_id, "HandleMatchFirstIntroducer");
}

auto HandleParseNode(Context& context,
                     Parse::MatchFirstDefinitionStartId node_id) -> bool {
  return context.TODO(node_id, "HandleMatchFirstDefinitionStart");
}

auto HandleParseNode(Context& context, Parse::MatchFirstId node_id) -> bool {
  return context.TODO(node_id, "HandleMatchFirst");
}

}  // namespace Carbon::Check
