// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_FIXED_SIZE_VALUE_STORE_H_
#define CARBON_TOOLCHAIN_BASE_FIXED_SIZE_VALUE_STORE_H_

#include "common/check.h"
#include "common/move_only.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/base/mem_usage.h"
#include "toolchain/base/value_store.h"

namespace Carbon {

// A value store with a predetermined size.
template <typename IdT, typename ValueT>
class FixedSizeValueStore : public MoveOnly<FixedSizeValueStore<IdT, ValueT>> {
 public:
  using ValueType = ValueStoreTypes<IdT, ValueT>::ValueType;
  using RefType = ValueStoreTypes<IdT, ValueT>::RefType;
  using ConstRefType = ValueStoreTypes<IdT, ValueT>::ConstRefType;

  // Makes a ValueStore of the specified size, but without initializing values.
  // Entries must be set before reading.
  static auto MakeForOverwrite(size_t size) -> FixedSizeValueStore {
    FixedSizeValueStore store;
    store.values_.resize_for_overwrite(size);
    return store;
  }

  // Makes a ValueStore of the specified size, initialized to a default.
  explicit FixedSizeValueStore(size_t size, ValueT default_value) {
    values_.resize(size, default_value);
  }

  // Sets the value for an ID.
  auto Set(IdT id, ValueType value) -> void {
    CARBON_DCHECK(id.index >= 0, "{0}", id);
    values_[id.index] = value;
  }

  // Returns a mutable value for an ID.
  auto Get(IdT id) -> RefType {
    CARBON_DCHECK(id.index >= 0, "{0}", id);
    return values_[id.index];
  }

  // Returns the value for an ID.
  auto Get(IdT id) const -> ConstRefType {
    CARBON_DCHECK(id.index >= 0, "{0}", id);
    return values_[id.index];
  }

  // Collects memory usage of the values.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Collect(label.str(), values_);
  }

  auto size() const -> size_t { return values_.size(); }

 private:
  // Allow default construction for `MakeForOverwrite`.
  FixedSizeValueStore() = default;

  // Storage for the `ValueT` objects, indexed by the id.
  llvm::SmallVector<ValueT, 0> values_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_FIXED_SIZE_VALUE_STORE_H_
