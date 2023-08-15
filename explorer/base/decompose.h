// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Utilities for types that support the `Decompose` API.

#ifndef CARBON_EXPLORER_BASE_DECOMPOSE_H_
#define CARBON_EXPLORER_BASE_DECOMPOSE_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"

namespace Carbon {

// CRTP base class which extends `Base` to support hashing and equality
// comparison. `Base` must support the `Decompose` API.
template <typename Base>
class HashFromDecompose {
 public:
  friend auto operator==(const HashFromDecompose& lhs,
                         const HashFromDecompose& rhs) -> bool {
    return static_cast<const Base*>(&lhs)->Decompose(
        [&](auto&&... lhs_elements) {
          return static_cast<const Base*>(&rhs)->Decompose(
              [&](auto&&... rhs_elements) {
                return ((lhs_elements == rhs_elements) && ...);
              });
        });
  }

  friend auto hash_value(const HashFromDecompose& self) -> llvm::hash_code {
    return static_cast<const Base*>(&self)->Decompose(
        [&](auto&&... lhs_elements) {
          return llvm::hash_combine(WrapForHash(lhs_elements)...);
        });
  }

 private:
  // Wraps T in a form that supports `hash_value`. Used for adapting types that
  // we can't extend directly.
  template <typename T>
  static auto WrapForHash(const T& t) -> const T& {
    return t;
  }

  template <typename T>
  static auto WrapForHash(const std::vector<T>& vec) -> llvm::ArrayRef<T> {
    return vec;
  }
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_BASE_DECOMPOSE_H_
