// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/handle.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::RequireIntroducerId node_id)
    -> bool {
  return context.TODO(node_id, "HandleRequireIntroducerId");
}

auto HandleParseNode(Context& context, Parse::ImplsId node_id) -> bool {
  return context.TODO(node_id, "HandleImplsId");
}

auto HandleParseNode(Context& context, Parse::RequireId node_id) -> bool {
  return context.TODO(node_id, "HandleRequireId");
}

}  // namespace Carbon::Check
