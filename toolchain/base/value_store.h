// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_VALUE_STORE_H_
#define CARBON_TOOLCHAIN_BASE_VALUE_STORE_H_

#include <limits>
#include <type_traits>
#include <utility>

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/mem_usage.h"
#include "toolchain/base/value_store_chunk.h"
#include "toolchain/base/value_store_types.h"
#include "toolchain/base/yaml.h"

namespace Carbon {

namespace Internal {

// Used as a parent class for non-printable types. This is just for
// std::conditional, not as an API.
class ValueStoreNotPrintable {};

}  // namespace Internal

template <class IdT>
class ValueStoreRange;

// A simple wrapper for accumulating values, providing IDs to later retrieve the
// value. This does not do deduplication.
//
// IdT::ValueType must represent the type being indexed.
template <typename IdT>
  requires(Internal::IdHasValueType<IdT>)
class ValueStore
    : public std::conditional<
          std::is_base_of_v<Printable<typename IdT::ValueType>,
                            typename IdT::ValueType>,
          Yaml::Printable<ValueStore<IdT>>, Internal::ValueStoreNotPrintable> {
 public:
  using IdType = IdT;
  using ValueType = ValueStoreTypes<IdT>::ValueType;
  using RefType = ValueStoreTypes<IdT>::RefType;
  using ConstRefType = ValueStoreTypes<IdT>::ConstRefType;

  ValueStore() = default;

  // Stores the value and returns an ID to reference it.
  auto Add(ValueType value) -> IdT {
    // This routine is especially hot and the check here relatively expensive
    // for the value provided, so only do this in non-optimized builds to make
    // tracking down issues easier.
    CARBON_DCHECK(size_ < std::numeric_limits<int32_t>::max(), "Id overflow");

    IdT id(size_);
    auto [chunk_index, pos] = Internal::IdToChunkIndices(id);
    ++size_;

    CARBON_DCHECK(static_cast<size_t>(chunk_index) <= chunks_.size(),
                  "{0} <= {1}", chunk_index, chunks_.size());
    if (static_cast<size_t>(chunk_index) == chunks_.size()) {
      chunks_.emplace_back();
    }

    CARBON_DCHECK(pos == chunks_[chunk_index].size());
    chunks_[chunk_index].push(std::move(value));
    return id;
  }

  // Returns a mutable value for an ID.
  auto Get(IdT id) -> RefType {
    CARBON_DCHECK(id.index >= 0, "{0}", id);
    CARBON_DCHECK(id.index < size_, "{0}", id);
    auto [chunk_index, pos] = Internal::IdToChunkIndices(id);
    return chunks_[chunk_index].at(pos);
  }

  // Returns the value for an ID.
  auto Get(IdT id) const -> ConstRefType {
    CARBON_DCHECK(id.index >= 0, "{0}", id);
    CARBON_DCHECK(id.index < size_, "{0}", id);
    auto [chunk_index, pos] = Internal::IdToChunkIndices(id);
    return chunks_[chunk_index].at(pos);
  }

  // Reserves space.
  auto Reserve(size_t size) -> void {
    // We get the number of chunks needed to satisfy `size` by rounding any
    // partial result up.
    size_t num_more_chunks =
        (size + ChunkType::Capacity - 1) / ChunkType::Capacity;
    if (chunks_.size() < num_more_chunks) {
      // We resize() rather than reserve() here to create the new `ChunkType`
      // objects, which will in turn allocate space for values in those chunks
      // (but not initialize them).
      chunks_.resize(num_more_chunks);
    }
  }

  // These are to support printable structures, and are not guaranteed.
  auto OutputYaml() const -> Yaml::OutputMapping {
    return Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
      for (auto [id, value] : enumerate()) {
        map.Add(PrintToString(id), Yaml::OutputScalar(value));
      }
    });
  }

  // Collects memory usage of the values.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Add(label.str(), size_ * sizeof(ValueType),
                  ChunkType::CapacityBytes * chunks_.size());
  }

  auto size() const -> size_t { return size_; }

  // Makes an iterable range over references to all values in the ValueStore.
  auto values() const [[clang::lifetimebound]] -> ValueStoreRange<IdT> {
    return ValueStoreRange<IdT>(*this);
  }

  // Makes an iterable range over pairs of the index and a reference to the
  // value for each value in the store.
  //
  // The range is over references to the values in the store, even if used with
  // `auto` to destructure the pair. In this example, the `value` is a
  // `ConstRefType`:
  // ```
  // for (auto [id, value] : store.enumerate()) { ... }
  // ```
  auto enumerate() const [[clang::lifetimebound]] -> auto {
    // For `it->val`, writing `const std::pair` is required; otherwise
    // `mapped_iterator` incorrectly infers the pointer type for `PointerProxy`.
    // NOLINTNEXTLINE(readability-const-return-type)
    auto index_to_id = [&](int32_t i) -> const std::pair<IdT, ConstRefType> {
      return std::pair<IdT, ConstRefType>(IdT(i), Get(IdT(i)));
    };
    // Because indices into `ValueStore` are all sequential values from 0, we
    // can use llvm::seq to walk all indices in the store.
    return llvm::map_range(llvm::seq(size_), index_to_id);
  }

 private:
  friend class ValueStoreRange<IdT>;

  using ChunkType = Internal::ValueStoreChunk<IdT, ValueType>;

  // Number of elements added to the store. The number should never exceed what
  // fits in an `int32_t`, which is checked in non-optimized builds in Add().
  int32_t size_ = 0;

  // Storage for the `ValueType` objects, indexed by the id. We use a vector of
  // chunks of `ValueType` instead of just a vector of `ValueType` so that
  // addresses of `ValueType` objects are stable. This allows the rest of the
  // toolchain to hold references into `ValueStore` without having to worry
  // about invalidation and use-after-free. We ensure at least one Chunk is held
  // inline so that in the case where there is only a single Chunk (i.e. small
  // files) we can avoid one indirection.
  llvm::SmallVector<ChunkType, 1> chunks_;
};

// A range over references to the values in a ValueStore, returned from
// `ValueStore::values()`. Hides the complex type name of the iterator
// internally to provide a type name (`ValueStoreRange<IdT>`) that can be
// referred to without auto and templates.
template <class IdT>
class ValueStoreRange {
 public:
  explicit ValueStoreRange(const ValueStore<IdT>& store
                           [[clang::lifetimebound]])
      : flattened_range_(MakeFlattenedRange(store)) {}

  auto begin() const -> auto { return flattened_range_.begin(); }
  auto end() const -> auto { return flattened_range_.end(); }

 private:
  // Flattens the range of `ValueStoreChunk`s of `ValueType`s into a single
  // range of `ValueType`s.
  static auto MakeFlattenedRange(const ValueStore<IdT>& store) -> auto {
    // Because indices into `ValueStore` are all sequential values from 0, we
    // can use llvm::seq to walk all indices in the store.
    return llvm::map_range(llvm::seq(store.size_),
                           [&](int32_t i) -> ValueStore<IdT>::ConstRefType {
                             return store.Get(IdT(i));
                           });
  }

  using FlattenedRangeType =
      decltype(MakeFlattenedRange(std::declval<const ValueStore<IdT>&>()));
  FlattenedRangeType flattened_range_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_VALUE_STORE_H_
