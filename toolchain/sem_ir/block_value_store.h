// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_BLOCK_VALUE_STORE_H_
#define CARBON_TOOLCHAIN_SEM_IR_BLOCK_VALUE_STORE_H_

#include <type_traits>

#include "common/set.h"
#include "toolchain/base/value_store.h"
#include "toolchain/base/yaml.h"

namespace Carbon::SemIR {

// Provides a block-based ValueStore, which uses slab allocation of added
// blocks. This allows references to values to outlast vector resizes that might
// otherwise invalidate references.
//
// BlockValueStore is used as-is, but there are also children that expose the
// protected members for type-specific functionality.
//
// On IdT, this requires:
//   - IdT::ElementType to represent the underlying type in the block.
//   - IdT::ValueType to be llvm::MutableArrayRef<IdT::ElementType> for
//     compatibility with ValueStore.
template <typename IdT>
class BlockValueStore : public Yaml::Printable<BlockValueStore<IdT>> {
 public:
  using ElementType = IdT::ElementType;

  explicit BlockValueStore(llvm::BumpPtrAllocator& allocator)
      : allocator_(&allocator) {
    auto empty = llvm::MutableArrayRef<ElementType>();
    auto empty_val = canonical_blocks_.Insert(
        empty, [&] { return values_.Add(empty); }, KeyContext(this));
    CARBON_CHECK(empty_val.key() == IdT::Empty);
  }

  // Adds a block with the given content, returning an ID to reference it.
  auto Add(llvm::ArrayRef<ElementType> content) -> IdT {
    if (content.empty()) {
      return IdT::Empty;
    }
    return values_.Add(AllocateCopy(content));
  }

  // Returns the requested block.
  auto Get(IdT id) const -> llvm::ArrayRef<ElementType> {
    return values_.Get(id);
  }

  // Returns a mutable view of the requested block. This operation should be
  // avoided where possible; we generally want blocks to be immutable once
  // created.
  auto GetMutable(IdT id) -> llvm::MutableArrayRef<ElementType> {
    return values_.Get(id);
  }

  // Adds a block or finds an existing canonical block with the given content,
  // and returns an ID to reference it.
  auto AddCanonical(llvm::ArrayRef<ElementType> content) -> IdT {
    if (content.empty()) {
      return IdT::Empty;
    }
    auto result = canonical_blocks_.Insert(
        content, [&] { return Add(content); }, KeyContext(this));
    return result.key();
  }

  // Promotes an existing block ID to a canonical block ID, or returns an
  // existing canonical block ID if the block was already added. The specified
  // block must not be modified after this point.
  auto MakeCanonical(IdT id) -> IdT {
    // Get the content first so that we don't have unnecessary translation of
    // the `id` into the content during insertion.
    auto result = canonical_blocks_.Insert(
        Get(id), [id] { return id; }, KeyContext(this));
    return result.key();
  }

  auto OutputYaml() const -> Yaml::OutputMapping {
    return Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
      for (auto [block_id, block] : values_.enumerate()) {
        map.Add(PrintToString(block_id),
                Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
                  for (auto [i, elem_id] : llvm::enumerate(block)) {
                    map.Add(llvm::itostr(i), Yaml::OutputScalar(elem_id));
                  }
                }));
      }
    });
  }

  // Collects memory usage of members.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Collect(MemUsage::ConcatLabel(label, "values_"), values_);
    mem_usage.Collect(MemUsage::ConcatLabel(label, "canonical_blocks_"),
                      canonical_blocks_, KeyContext(this));
  }

  auto size() const -> int { return values_.size(); }

 protected:
  // Allocates a copy of the given data using our slab allocator.
  auto AllocateCopy(llvm::ArrayRef<ElementType> data)
      -> llvm::MutableArrayRef<ElementType> {
    auto result = AllocateUninitialized(data.size());
    std::uninitialized_copy(data.begin(), data.end(), result.begin());
    return result;
  }

  // Allocates an uninitialized array using our slab allocator.
  auto AllocateUninitialized(size_t size)
      -> llvm::MutableArrayRef<ElementType> {
    // We're not going to run a destructor, so ensure that's OK.
    static_assert(std::is_trivially_destructible_v<ElementType>);

    auto storage = static_cast<ElementType*>(
        allocator_->Allocate(size * sizeof(ElementType), alignof(ElementType)));
    return llvm::MutableArrayRef<ElementType>(storage, size);
  }

  // Allow children to have more complex value handling.
  auto values() -> ValueStore<IdT>& { return values_; }

 private:
  class KeyContext;

  llvm::BumpPtrAllocator* allocator_;
  ValueStore<IdT> values_;
  Set<IdT, /*SmallSize=*/0, KeyContext> canonical_blocks_;
};

template <typename IdT>
class BlockValueStore<IdT>::KeyContext
    : public TranslatingKeyContext<KeyContext> {
 public:
  explicit KeyContext(const BlockValueStore* store) : store_(store) {}

  auto TranslateKey(IdT id) const -> llvm::ArrayRef<ElementType> {
    return store_->Get(id);
  }

 private:
  const BlockValueStore* store_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_BLOCK_VALUE_STORE_H_
