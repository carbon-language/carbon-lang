// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_DUMP_ID_H_
#define CARBON_TOOLCHAIN_CHECK_DUMP_ID_H_

#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

class Context;

auto DumpIdImpl(const Context& context, SemIR::LocId loc_id) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DUMP_ID_H_
