// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/cpp_global_var.h"

#include "toolchain/base/canonical_value_store_impl.h"
#include "toolchain/base/value_store_impl.h"

namespace Carbon {
template class CanonicalValueStore<SemIR::CppGlobalVarId,
                                   SemIR::CppGlobalVarKey,
                                   Tag<SemIR::CheckIRId>, SemIR::CppGlobalVar>;
template class ValueStore<SemIR::CppGlobalVarId, SemIR::CppGlobalVar,
                          Tag<SemIR::CheckIRId>>;
}  // namespace Carbon
