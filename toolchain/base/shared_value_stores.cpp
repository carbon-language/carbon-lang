// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/shared_value_stores.h"

#include "toolchain/base/canonical_value_store_impl.h"
#include "toolchain/base/value_store_impl.h"

namespace Carbon {

template class CanonicalValueStore<FloatId, llvm::APFloat>;
template class CanonicalValueStore<IdentifierId, llvm::StringRef>;
template class CanonicalValueStore<StringLiteralValueId, llvm::StringRef>;
template class ValueStore<FloatId, llvm::APFloat>;
template class ValueStore<IdentifierId, llvm::StringRef>;
template class ValueStore<RealId, Real>;
template class ValueStore<StringLiteralValueId, llvm::StringRef>;

}  // namespace Carbon
