// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_HASHING_LLVM_H_
#define CARBON_COMMON_HASHING_LLVM_H_

#include "common/hashing.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Hashing.h"

namespace Carbon::InternalHashDispatch {

template <>
struct CustomHashValue<llvm::APInt> {
  static auto Hash(llvm::APInt value, uint64_t seed) -> HashCode {
    Hasher hasher(seed);
    if (LLVM_LIKELY(value.isSingleWord())) {
      hasher.Hash(value.getBitWidth(), value.getZExtValue());
    } else {
      hasher.HashRaw(value.getBitWidth());
      hasher.HashSizedBytes(
          llvm::ArrayRef(value.getRawData(), value.getNumWords()));
    }
    return static_cast<HashCode>(hasher);
  }
};

template <>
struct CustomHashValue<llvm::APFloat> {
  static auto Hash(llvm::APFloat value, uint64_t seed) -> HashCode {
    Hasher hasher(seed);
    // Hashing floating point numbers is complex and depends on the specific
    // internal semantics of `APFloat`, so delegate to the LLVM hashing
    // framework here. We re-hash the result to mix in our seed. All of this is
    // a bit inefficient, and we can revisit this to provide a dedicated
    // implementation if it becomes a bottleneck.
    using llvm::hash_value;
    hasher.HashRaw(hash_value(value));
    return static_cast<HashCode>(hasher);
  }
};

}  // namespace Carbon::InternalHashDispatch

#endif  // CARBON_COMMON_HASHING_LLVM_H_
