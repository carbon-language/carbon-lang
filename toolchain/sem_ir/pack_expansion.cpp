// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/pack_expansion.h"

#include "toolchain/base/value_store_impl.h"

namespace Carbon {
template class ValueStore<SemIR::PackExpansionId, SemIR::PackExpansion,
                          Tag<SemIR::CheckIRId>>;
}  // namespace Carbon
