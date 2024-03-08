// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleFileStart(Context& /*context*/, Parse::FileStartId /*node_id*/)
    -> bool {
  // No action to perform.
  // TODO: We may want to push `FileStart` as a sentinel so that `Peek`s can't
  // fail.
  return true;
}

auto HandleFileEnd(Context& /*context*/, Parse::FileEndId /*node_id*/) -> bool {
  // No action to perform.
  return true;
}

}  // namespace Carbon::Check
