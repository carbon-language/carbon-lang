// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/observe.h"

#include "toolchain/base/block_value_store_impl.h"
#include "toolchain/base/value_store_impl.h"

namespace Carbon {
template class ValueStore<SemIR::ObserveId, SemIR::Observe,
                          Tag<SemIR::CheckIRId>>;
template class ValueStore<SemIR::ObserveBlockId,
                          llvm::MutableArrayRef<SemIR::ObserveId>,
                          Tag<SemIR::CheckIRId>>;
template class BlockValueStore<SemIR::ObserveBlockId, SemIR::ObserveId,
                               Tag<SemIR::CheckIRId>>;
}  // namespace Carbon
