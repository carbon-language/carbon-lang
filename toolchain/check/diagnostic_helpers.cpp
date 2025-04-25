// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/diagnostic_helpers.h"

#include "toolchain/check/context.h"

namespace Carbon::Check {
class Context;

auto TokenOnly(Context& /*context*/, Parse::NodeId node_id) -> SemIR::LocId {
  return SemIR::LocId(node_id).ToTokenOnly();
}

auto TokenOnly(Context& context, SemIR::LocId loc_id) -> SemIR::LocId {
  return context.insts().GetCanonicalLocId(loc_id).ToTokenOnly();
}

}  // namespace Carbon::Check
