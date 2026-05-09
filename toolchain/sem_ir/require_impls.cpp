// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/require_impls.h"

#include "toolchain/base/block_value_store_impl.h"
#include "toolchain/base/value_store_impl.h"

namespace Carbon {
template class ValueStore<SemIR::RequireImplsId, SemIR::RequireImpls,
                          Tag<SemIR::CheckIRId>>;
template class ValueStore<SemIR::RequireImplsBlockId,
                          llvm::MutableArrayRef<SemIR::RequireImplsId>,
                          Tag<SemIR::CheckIRId>>;
template class BlockValueStore<SemIR::RequireImplsBlockId,
                               SemIR::RequireImplsId, Tag<SemIR::CheckIRId>>;
}  // namespace Carbon
