// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_VALUE_STORE_IMPL_H_
#define CARBON_TOOLCHAIN_BASE_VALUE_STORE_IMPL_H_

#include "toolchain/base/value_store.h"

namespace Carbon {

template <typename IdT, typename ValueT, typename TagIdT>
ValueStore<IdT, ValueT, TagIdT>::ValueStore()
  requires(IdTagIsUntagged<IdTagType>)
= default;

template <typename IdT, typename ValueT, typename TagIdT>
ValueStore<IdT, ValueT, TagIdT>::ValueStore(IdTagType tag)
  requires(!IdTagIsUntagged<IdTagType>)
    : tag_(tag) {}

template <typename IdT, typename ValueT, typename TagIdT>
ValueStore<IdT, ValueT, TagIdT>::ValueStore(
    typename ValueStore<IdT, ValueT, TagIdT>::IdTagType::TagIdType id,
    int32_t initial_reserved_ids)
  requires(!IdTagIsUntagged<IdTagType>)
    : tag_(id, initial_reserved_ids) {}

template <typename IdT, typename ValueT, typename TagIdT>
ValueStore<IdT, ValueT, TagIdT>::~ValueStore() = default;

template <typename IdT, typename ValueT, typename TagIdT>
ValueStore<IdT, ValueT, TagIdT>::ValueStore(ValueStore&&) noexcept = default;

template <typename IdT, typename ValueT, typename TagIdT>
auto ValueStore<IdT, ValueT, TagIdT>::operator=(ValueStore&&) noexcept
    -> ValueStore& = default;

template <typename IdT, typename ValueT, typename TagIdT>
auto ValueStore<IdT, ValueT, TagIdT>::Resize(int32_t size,
                                             ConstRefType default_value) -> void
  requires(std::is_copy_constructible_v<ValueT>)
{
  if (size <= size_) {
    return;
  }

  auto [begin_chunk_index, begin_pos] = RawIndexToChunkIndices(size_);
  // Use an inclusive range so that if `size` would be the next chunk, we
  // don't try doing something with it.
  auto [end_chunk_index, end_pos] = RawIndexToChunkIndices(size - 1);
  chunks_.resize(end_chunk_index + 1);

  // If the begin and end chunks are the same, we only fill from begin to end.
  if (begin_chunk_index == end_chunk_index) {
    chunks_[begin_chunk_index].UninitializedFill(end_pos - begin_pos + 1,
                                                 default_value);
  } else {
    // Otherwise, we do partial fills on the begin and end chunk, and full
    // fills on intermediate chunks.
    chunks_[begin_chunk_index].UninitializedFill(Chunk::Capacity() - begin_pos,
                                                 default_value);
    for (auto i = begin_chunk_index + 1; i < end_chunk_index; ++i) {
      chunks_[i].UninitializedFill(Chunk::Capacity(), default_value);
    }
    chunks_[end_chunk_index].UninitializedFill(end_pos + 1, default_value);
  }

  // Update size.
  size_ = size;
}

template <typename IdT, typename ValueT, typename TagIdT>
auto ValueStore<IdT, ValueT, TagIdT>::Chunk::UninitializedFill(
    int32_t fill_count,
    typename ValueStore<IdT, ValueT, TagIdT>::ConstRefType default_value)
    -> void
  requires(std::is_copy_constructible_v<ValueT>)
{
  CARBON_DCHECK(num_ + fill_count <= Capacity());
  std::uninitialized_fill_n(buf_ + num_, fill_count, default_value);
  num_ += fill_count;
}

template <typename IdT, typename ValueT, typename TagIdT>
auto ValueStore<IdT, ValueT, TagIdT>::OutputYaml() const
    -> Yaml::OutputMapping {
  return Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
    for (auto [id, value] : enumerate()) {
      if constexpr (requires(llvm::raw_ostream& out) { out << value; }) {
        map.Add(PrintToString(id), Yaml::OutputScalar(value));
      } else {
        map.Add(PrintToString(id), Yaml::OutputScalar("<unprintable>"));
      }
    }
  });
}

template <typename IdT, typename ValueT, typename TagIdT>
auto ValueStore<IdT, ValueT, TagIdT>::CollectMemUsage(
    MemUsage& mem_usage, llvm::StringRef label) const -> void {
  mem_usage.Add(label.str(), size_ * sizeof(ValueType),
                Chunk::CapacityBytes * chunks_.size());
}

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_VALUE_STORE_IMPL_H_
