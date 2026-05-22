// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/specific_interface.h"

#include "toolchain/base/canonical_value_store_impl.h"
#include "toolchain/base/value_store_impl.h"

namespace Carbon {
template class CanonicalValueStore<SemIR::SpecificInterfaceId,
                                   SemIR::SpecificInterface,
                                   Tag<SemIR::CheckIRId>>;
template class ValueStore<SemIR::SpecificInterfaceId, SemIR::SpecificInterface,
                          Tag<SemIR::CheckIRId>>;
}  // namespace Carbon
