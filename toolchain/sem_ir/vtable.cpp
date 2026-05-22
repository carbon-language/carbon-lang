// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/vtable.h"

#include "toolchain/base/value_store_impl.h"

namespace Carbon {
template class ValueStore<SemIR::VtableId, SemIR::Vtable,
                          Tag<SemIR::CheckIRId>>;
}  // namespace Carbon
