// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_GROUPED_VALUE_STORE_H_
#define CARBON_TOOLCHAIN_BASE_GROUPED_VALUE_STORE_H_

#include <concepts>
#include <cstddef>
#include <cstdint>

#include "common/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/id_tag.h"

namespace Carbon {

// A fixed collection of values, grouped by an ID: for example, the specifics of
// each generic, or the instructions checked from each token.
//
// The values are held in a single array, ordered by the ID of the group they
// belong to, alongside the index at which each group starts. Because IDs are
// dense, that layout can be built by a counting sort in two passes over the
// values, and a group can then be found in constant time. A map from ID to a
// list of values would instead hash on every lookup and allocate per group.
//
// The trade-off is that the contents are fixed once built: a group is a
// contiguous range of the value array, so nothing can be added to it later.
template <typename IdT, typename ValueT, typename TagIdT = Untagged>
class GroupedValueStore {
 public:
  using IdType = IdT;
  using IdTagType = IdTag<IdT, TagIdT>;

  // Builds the groups for the IDs vended by `id_source`, which is the value
  // store that the group IDs index.
  //
  // `enumerate` is called with a function to call with each (ID, value) pair to
  // store; pairs whose ID has no value are ignored. Note that `enumerate` is
  // called more than once, and must produce the same pairs each time.
  template <typename ValueStoreT, typename EnumerateFn>
    requires(std::same_as<IdT, typename ValueStoreT::IdType> &&
             !IdTagIsUntagged<IdTagType>)
  explicit GroupedValueStore(const ValueStoreT& id_source,
                             EnumerateFn enumerate)
      : GroupedValueStore(id_source.size(), id_source.GetIdTag(), enumerate) {}

  // Builds the groups for the `num_ids` IDs with indexes `[0, num_ids)`.
  template <typename EnumerateFn>
    requires(IdTagIsUntagged<IdTagType>)
  explicit GroupedValueStore(size_t num_ids, EnumerateFn enumerate)
      : GroupedValueStore(num_ids, IdTagType(), enumerate) {}

  // Returns the values in the group for `id`, in the order they were
  // enumerated. This is empty if no value was added for `id`, including when
  // `id` has no value.
  auto Get(IdT id) const -> llvm::ArrayRef<ValueT> {
    if (!id.has_value()) {
      return {};
    }
    int32_t index = tag_.Remove(id);
    CARBON_CHECK(static_cast<size_t>(index) + 1 < starts_.size(),
                 "{0} is not an ID of a group in this store", id);
    return llvm::ArrayRef(values_).slice(starts_[index],
                                         starts_[index + 1] - starts_[index]);
  }

  auto size() const -> size_t { return values_.size(); }

 private:
  template <typename EnumerateFn>
  explicit GroupedValueStore(size_t num_ids, IdTagType tag,
                             EnumerateFn enumerate);

  // The values of every group, ordered by group: the group for the ID with
  // index `i` is `values_[starts_[i] .. starts_[i + 1])`, so that `starts_` has
  // one more element than there are IDs.
  llvm::SmallVector<ValueT, 0> values_;
  llvm::SmallVector<int32_t, 0> starts_;

  IdTagType tag_;
};

template <typename IdT, typename ValueT, typename TagIdT>
template <typename EnumerateFn>
GroupedValueStore<IdT, ValueT, TagIdT>::GroupedValueStore(size_t num_ids,
                                                          IdTagType tag,
                                                          EnumerateFn enumerate)
    : tag_(tag) {
  // The index of the group that `id` belongs to. This is checked rather than
  // assumed because an out-of-range index would write outside `starts_`.
  auto group_index = [&](IdT id) {
    int32_t index = tag_.Remove(id);
    CARBON_CHECK(static_cast<size_t>(index) < num_ids,
                 "{0} is not an ID of a group in this store", id);
    return index;
  };

  // Populate `starts_` in three in-place passes. Note that we need N+1 elements
  // to hold the boundaries of N contiguous groups, plus an additional temporary
  // element for reasons discussed below.
  //
  // First, we count the values per ID. The array contents are shifted by 2:
  // `starts_[i + 2]` will hold the number of values for the ID with `.index ==
  // i`.
  starts_.assign(num_ids + 2, 0);
  enumerate([&](IdT id, const ValueT& /*value*/) {
    if (id.has_value()) {
      ++starts_[group_index(id) + 2];
    }
  });

  // Perform a prefix sum, so that `starts_[i + 2]` holds the number of values
  // for IDs with `.index <= i`, that is, the end of the group for ID `i`, and
  // hence `starts_[i + 1]` is the start of the group for ID `i`.
  for (size_t i = 1; i < starts_.size(); ++i) {
    starts_[i] += starts_[i - 1];
  }

  // Pop the final "start" index, which is now the total number of values in all
  // the groups.
  int32_t num_values = starts_.pop_back_val();

  // Every element is written below, but the array still has to be filled with
  // something first, and an ID type has no default constructor.
  if constexpr (requires { ValueT::None; }) {
    values_.resize(num_values, ValueT::None);
  } else {
    values_.resize(num_values);
  }

  // Populate `values_`, using `starts_[i + 1]` as the index to write the next
  // value for ID `i`, which is incremented on each write. Thus, at the end of
  // the loop, `starts_[i + 1]` is the past-the-end index for ID `i`, that is,
  // the start index for ID `i + 1`, which is the final state of `starts_`.
  enumerate([&](IdT id, const ValueT& value) {
    if (id.has_value()) {
      values_[starts_[group_index(id) + 1]++] = value;
    }
  });

  CARBON_CHECK(static_cast<size_t>(starts_.back()) == values_.size(),
               "Enumeration produced a different number of values each time");
}

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_GROUPED_VALUE_STORE_H_
