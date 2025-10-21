// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/handle.h"
#include "toolchain/parse/node_ids.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::RequireIntroducerId node_id)
    -> bool {
  return context.TODO(node_id, "require");
}

auto HandleParseNode(Context& context, Parse::RequireDefaultSelfImplsId node_id)
    -> bool {
  return context.TODO(node_id, "require");
}

auto HandleParseNode(Context& context, Parse::RequireTypeImplsId node_id)
    -> bool {
  return context.TODO(node_id, "require");
}

auto HandleParseNode(Context& context, Parse::RequireDeclId node_id) -> bool {
  return context.TODO(node_id, "require");
}

}  // namespace Carbon::Check
