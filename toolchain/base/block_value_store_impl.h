// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_BLOCK_VALUE_STORE_IMPL_H_
#define CARBON_TOOLCHAIN_BASE_BLOCK_VALUE_STORE_IMPL_H_

#include "toolchain/base/block_value_store.h"

namespace Carbon {

template <typename IdT, typename ElementT, typename TagIdT>
BlockValueStore<IdT, ElementT, TagIdT>::BlockValueStore(
    llvm::BumpPtrAllocator& allocator, IdTagType::TagIdType tag_id,
    int32_t initial_reserved_ids)
  requires(!IdTagIsUntagged<IdTagType>)
    : allocator_(&allocator), values_(tag_id, initial_reserved_ids) {
  if constexpr (requires { IdT::Empty; }) {
    auto empty = RefType();
    auto empty_val = canonical_blocks_.Insert(
        empty, [&] { return values_.Add(empty); }, KeyContext(this));
    CARBON_CHECK(empty_val.key() == IdT::Empty);
  }
}

template <typename IdT, typename ElementT, typename TagIdT>
BlockValueStore<IdT, ElementT, TagIdT>::~BlockValueStore() = default;

template <typename IdT, typename ElementT, typename TagIdT>
BlockValueStore<IdT, ElementT, TagIdT>::BlockValueStore(
    BlockValueStore&&) noexcept = default;

template <typename IdT, typename ElementT, typename TagIdT>
auto BlockValueStore<IdT, ElementT, TagIdT>::operator=(
    BlockValueStore&&) noexcept -> BlockValueStore& = default;

template <typename IdT, typename ElementT, typename TagIdT>
auto BlockValueStore<IdT, ElementT, TagIdT>::OutputYaml() const
    -> Yaml::OutputMapping {
  return Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
    for (auto [block_id, block] : values_.enumerate()) {
      map.Add(PrintToString(block_id),
              Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
                for (auto [i, elem_id] : llvm::enumerate(block)) {
                  if constexpr (requires(llvm::raw_ostream& out) {
                                  out << elem_id;
                                }) {
                    map.Add(llvm::itostr(i), Yaml::OutputScalar(elem_id));
                  } else {
                    map.Add(llvm::itostr(i),
                            Yaml::OutputScalar("<unprintable>"));
                  }
                }
              }));
    }
  });
}

template <typename IdT, typename ElementT, typename TagIdT>
auto BlockValueStore<IdT, ElementT, TagIdT>::CollectMemUsage(
    MemUsage& mem_usage, llvm::StringRef label) const -> void {
  mem_usage.Collect(MemUsage::ConcatLabel(label, "values_"), values_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "canonical_blocks_"),
                    canonical_blocks_, KeyContext(this));
}

template <typename IdT, typename ElementT, typename TagIdT>
auto BlockValueStore<IdT, ElementT, TagIdT>::AllocateCopy(ConstRefType data)
    -> RefType {
  auto result = AllocateUninitialized(data.size());
  std::uninitialized_copy(data.begin(), data.end(), result.begin());
  return result;
}

template <typename IdT, typename ElementT, typename TagIdT>
auto BlockValueStore<IdT, ElementT, TagIdT>::AllocateUninitialized(size_t size)
    -> RefType {
  // We're not going to run a destructor, so ensure that's OK.
  static_assert(std::is_trivially_destructible_v<ElementType>);

  auto storage = static_cast<ElementType*>(
      allocator_->Allocate(size * sizeof(ElementType), alignof(ElementType)));
  return RefType(storage, size);
}

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_BLOCK_VALUE_STORE_IMPL_H_
