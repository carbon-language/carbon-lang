// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_CANONICAL_VALUE_STORE_IMPL_H_
#define CARBON_TOOLCHAIN_BASE_CANONICAL_VALUE_STORE_IMPL_H_

#include "toolchain/base/canonical_value_store.h"

namespace Carbon {

template <typename IdT, typename KeyT, typename TagIdT, typename ValueT>
CanonicalValueStore<IdT, KeyT, TagIdT, ValueT>::CanonicalValueStore()
  requires(IdTagIsUntagged<IdTag<IdT, TagIdT>>)
= default;

template <typename IdT, typename KeyT, typename TagIdT, typename ValueT>
CanonicalValueStore<IdT, KeyT, TagIdT, ValueT>::CanonicalValueStore(
    typename IdTagType::TagIdType id, int32_t initial_reserved_ids)
  requires(!IdTagIsUntagged<IdTag<IdT, TagIdT>>)
    : values_(id, initial_reserved_ids) {}

template <typename IdT, typename KeyT, typename TagIdT, typename ValueT>
CanonicalValueStore<IdT, KeyT, TagIdT, ValueT>::~CanonicalValueStore() =
    default;

template <typename IdT, typename KeyT, typename TagIdT, typename ValueT>
CanonicalValueStore<IdT, KeyT, TagIdT, ValueT>::CanonicalValueStore(
    CanonicalValueStore&&) noexcept = default;

template <typename IdT, typename KeyT, typename TagIdT, typename ValueT>
auto CanonicalValueStore<IdT, KeyT, TagIdT, ValueT>::operator=(
    CanonicalValueStore&&) noexcept -> CanonicalValueStore& = default;

template <typename IdT, typename KeyT, typename TagIdT, typename ValueT>
auto CanonicalValueStore<IdT, KeyT, TagIdT, ValueT>::OutputYaml() const
    -> Yaml::OutputMapping {
  return values_.OutputYaml();
}

template <typename IdT, typename KeyT, typename TagIdT, typename ValueT>
auto CanonicalValueStore<IdT, KeyT, TagIdT, ValueT>::CollectMemUsage(
    MemUsage& mem_usage, llvm::StringRef label) const -> void {
  mem_usage.Collect(MemUsage::ConcatLabel(label, "values_"), values_);
  auto bytes = set_.ComputeMetrics(KeyContext(&values_)).storage_bytes;
  mem_usage.Add(MemUsage::ConcatLabel(label, "set_"), bytes, bytes);
}

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_CANONICAL_VALUE_STORE_IMPL_H_
