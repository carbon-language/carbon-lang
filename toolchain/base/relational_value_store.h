// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_RELATIONAL_VALUE_STORE_H_
#define CARBON_TOOLCHAIN_BASE_RELATIONAL_VALUE_STORE_H_

#include <optional>

#include "common/check.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/value_store_types.h"

namespace Carbon {

// A ValueStore that builds a 1:1 relationship between two IDs.
// * `RelatedIdT` represents a related ID that can be used to find values in the
//   store.
// * `IdT` is the actual ID of values in this store, and `IdT::ValueType` is the
//   value type being stored.
//
// The value store builds a mapping so that either ID can be used later to find
// a value. And the user can query if a related `RelatedIdT` has been used to
// add a value to the store or not.
//
// When adding to the store, the user provides the related `RelatedIdT` along
// with the value being stored, and gets back the ID of the value in the store.
//
// This store requires more storage space than normal ValueStore does, as it
// requires storing a bit for presence of each `RelatedIdT`. And it allocates
// memory for values for all IDs up largest ID present in the store, even if
// they are not yet used.
template <typename RelatedIdT, typename IdT, typename ValueT>
class RelationalValueStore {
 public:
  using ValueType = ValueStoreTypes<ValueT>::ValueType;
  using ConstRefType = ValueStoreTypes<ValueT>::ConstRefType;

  // Given the related ID and a value, stores the value and returns a mapped ID
  // to reference it in the store.
  auto Add(RelatedIdT related_id, ValueType value) -> IdT {
    CARBON_DCHECK(related_id.index >= 0, "{0}", related_id);
    IdT id(related_id.index);
    if (static_cast<size_t>(id.index) >= values_.size()) {
      values_.resize(id.index + 1);
    }
    auto& opt = values_[id.index];
    CARBON_CHECK(!opt.has_value(),
                 "Add with `related_id` that was already added to the store");
    opt.emplace(std::move(value));
    return id;
  }

  // Returns the ID of a value in the store if the `related_id` was previously
  // used to add a value to the store, or None.
  auto TryGetId(RelatedIdT related_id) const -> IdT {
    CARBON_DCHECK(related_id.index >= 0, "{0}", related_id);
    if (static_cast<size_t>(related_id.index) >= values_.size()) {
      return IdT::None;
    }
    auto& opt = values_[related_id.index];
    if (!opt.has_value()) {
      return IdT::None;
    }
    return IdT(related_id.index);
  }

  // Returns a value for an ID.
  auto Get(IdT id) const -> ConstRefType {
    CARBON_DCHECK(id.index >= 0, "{0}", id);
    return *values_[id.index];
  }

 private:
  // Set inline size to 0 because these will typically be too large for the
  // stack, while this does make File smaller.
  llvm::SmallVector<std::optional<ValueType>, 0> values_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_RELATIONAL_VALUE_STORE_H_
