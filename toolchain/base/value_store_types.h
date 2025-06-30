// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_VALUE_STORE_TYPES_H_
#define CARBON_TOOLCHAIN_BASE_VALUE_STORE_TYPES_H_

#include <concepts>
#include <type_traits>

#include "llvm/ADT/StringRef.h"

namespace Carbon {

// Common calculation for ValueStore types.
template <typename IdT, typename ValueT = IdT::ValueType,
          typename KeyT = ValueT>
class ValueStoreTypes {
 public:
  using ValueType = std::decay_t<ValueT>;

  // TODO: Would be a bit cleaner to not have this here as it's only meaningful
  // to the `CanonicalValueStore`, not to other `ValueStore`s. Planned to fix
  // with a larger refactoring.
  using KeyType = std::decay_t<KeyT>;

  // Typically we want to use `ValueType&` and `const ValueType& to avoid
  // copies, but when the value type is a `StringRef`, we assume external
  // storage for the string data and both our value type and ref type will be
  // `StringRef`. This will preclude mutation of the string data.
  using RefType = std::conditional_t<std::same_as<llvm::StringRef, ValueType>,
                                     llvm::StringRef, ValueType&>;
  using ConstRefType =
      std::conditional_t<std::same_as<llvm::StringRef, ValueType>,
                         llvm::StringRef, const ValueType&>;
};

// If `IdT` provides a distinct `IdT::KeyType`, default to that for the key
// type.
template <typename IdT>
  requires(!std::same_as<typename IdT::ValueType, typename IdT::KeyType>)
class ValueStoreTypes<IdT>
    : public ValueStoreTypes<IdT, typename IdT::ValueType,
                             typename IdT::KeyType> {};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_VALUE_STORE_TYPES_H_
