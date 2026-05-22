// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/handle.h"
#include "toolchain/parse/node_ids.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::ObserveIntroducerId node_id)
    -> bool {
  return context.TODO(node_id, "ObserveIntroducerId");
}

auto HandleParseNode(Context& context, Parse::ObserveEqualEqualId node_id)
    -> bool {
  return context.TODO(node_id, "ObserveEqualEqualId");
}

auto HandleParseNode(Context& context, Parse::ObserveImplsId node_id) -> bool {
  return context.TODO(node_id, "ObserveImplsId");
}

auto HandleParseNode(Context& context, Parse::ObserveDeclId node_id) -> bool {
  return context.TODO(node_id, "ObserveDeclId");
}

}  // namespace Carbon::Check
