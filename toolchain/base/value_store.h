// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_VALUE_STORE_H_
#define CARBON_TOOLCHAIN_BASE_VALUE_STORE_H_

#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>

#include "common/check.h"
#include "common/hashtable_key_context.h"
#include "common/ostream.h"
#include "common/set.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Compiler.h"
#include "toolchain/base/mem_usage.h"
#include "toolchain/base/value_store_chunk.h"
#include "toolchain/base/yaml.h"

namespace Carbon {

namespace Internal {

// Used as a parent class for non-printable types. This is just for
// std::conditional, not as an API.
class ValueStoreNotPrintable {};

}  // namespace Internal

template <class IdT>
class ValueStoreRange;

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

// A wrapper for accumulating immutable values with deduplication, providing IDs
// to later retrieve the value.
//
// `IdT::ValueType` must represent the type being indexed.
//
// `IdT::KeyType` can optionally be present, and if so is used for the argument
// to `Lookup`. It must be valid to use both `KeyType` and `ValueType` as lookup
// types in the underlying `Set`.
template <typename IdT>
class CanonicalValueStore {
 public:
  using ValueType = ValueStoreTypes<IdT>::ValueType;
  using KeyType = ValueStoreTypes<IdT>::KeyType;
  using RefType = ValueStoreTypes<IdT>::RefType;
  using ConstRefType = ValueStoreTypes<IdT>::ConstRefType;

  // Stores a canonical copy of the value and returns an ID to reference it.
  auto Add(ValueType value) -> IdT;

  // Returns the value for an ID.
  auto Get(IdT id) const -> ConstRefType { return values_.Get(id); }

  // Looks up the canonical ID for a value, or returns `None` if not in the
  // store.
  auto Lookup(KeyType key) const -> IdT;

  // Reserves space.
  auto Reserve(size_t size) -> void;

  // These are to support printable structures, and are not guaranteed.
  auto OutputYaml() const -> Yaml::OutputMapping {
    return values_.OutputYaml();
  }

  auto values() const [[clang::lifetimebound]] -> ValueStoreRange<IdT> {
    return values_.values();
  }
  auto size() const -> size_t { return values_.size(); }

  // Collects memory usage of the values and deduplication set.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Collect(MemUsage::ConcatLabel(label, "values_"), values_);
    auto bytes = set_.ComputeMetrics(KeyContext(&values_)).storage_bytes;
    mem_usage.Add(MemUsage::ConcatLabel(label, "set_"), bytes, bytes);
  }

 private:
  class KeyContext;

  ValueStore<IdT> values_;
  Set<IdT, /*SmallSize=*/0, KeyContext> set_;
};

template <typename IdT>
class CanonicalValueStore<IdT>::KeyContext
    : public TranslatingKeyContext<KeyContext> {
 public:
  explicit KeyContext(const ValueStore<IdT>* values) : values_(values) {}

  // Note that it is safe to return a `const` reference here as the underlying
  // object's lifetime is provided by the `ValueStore`.
  auto TranslateKey(IdT id) const -> ValueStore<IdT>::ConstRefType {
    return values_->Get(id);
  }

 private:
  const ValueStore<IdT>* values_;
};

template <typename IdT>
auto CanonicalValueStore<IdT>::Add(ValueType value) -> IdT {
  auto make_key = [&] { return IdT(values_.Add(std::move(value))); };
  return set_.Insert(value, make_key, KeyContext(&values_)).key();
}

template <typename IdT>
auto CanonicalValueStore<IdT>::Lookup(KeyType key) const -> IdT {
  if (auto result = set_.Lookup(key, KeyContext(&values_))) {
    return result.key();
  }
  return IdT::None;
}

template <typename IdT>
auto CanonicalValueStore<IdT>::Reserve(size_t size) -> void {
  // Compute the resulting new insert count using the size of values -- the
  // set doesn't have a fast to compute current size.
  if (size > values_.size()) {
    set_.GrowForInsertCount(size - values_.size(), KeyContext(&values_));
  }
  values_.Reserve(size);
}

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
template <typename RelatedIdT, typename IdT>
class RelationalValueStore {
 public:
  using ValueType = ValueStoreTypes<IdT>::ValueType;
  using ConstRefType = ValueStoreTypes<IdT>::ConstRefType;

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
  llvm::SmallVector<std::optional<std::decay_t<ValueType>>, 0> values_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_VALUE_STORE_H_
