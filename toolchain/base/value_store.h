// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_VALUE_STORE_H_
#define CARBON_TOOLCHAIN_BASE_VALUE_STORE_H_

#include <bit>
#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemAlloc.h"
#include "toolchain/base/mem_usage.h"
#include "toolchain/base/value_store_types.h"
#include "toolchain/base/yaml.h"

namespace Carbon {

namespace Internal {

// Used as a parent class for non-printable types. This is just for
// std::conditional, not as an API.
class ValueStoreNotPrintable {};

}  // namespace Internal

struct IdTag {
  IdTag()
      : id_tag_(0),
        initial_reserved_ids_(std::numeric_limits<int32_t>::max()) {}
  explicit IdTag(int32_t id_index, int32_t initial_reserved_ids)
      : initial_reserved_ids_(initial_reserved_ids) {
    // Shift down by 1 to get out of the high bit to avoid using any negative
    // ids, since they have special uses.
    // Shift down by another 1 to free up the second highest bit for a marker to
    // indicate whether the index is tagged (& needs to be untagged) or not.
    // Add one to the index so it's not zero-based, to make it a bit less likely
    // this doesn't collide with anything else (though with the
    // second-highest-bit-tagging this might not be needed).
    id_tag_ = llvm::reverseBits((((id_index + 1) << 1) | 1) << 1);
  }
  auto GetCheckIRId() const -> int32_t {
    return (llvm::reverseBits(id_tag_) >> 2) - 1;
  }
  auto Apply(int32_t index) const -> int32_t {
    if (index < initial_reserved_ids_) {
      return index;
    }
    // assert that id_tag_ doesn't have the second highest bit set
    return index ^ id_tag_;
  }
  static auto DecomposeWithBestEffort(int32_t tagged_index)
      -> std::pair<int32_t, int32_t> {
    if (tagged_index < 0) {
      return {-1, tagged_index};
    }
    if ((llvm::reverseBits(2) & tagged_index) == 0) {
      return {-1, tagged_index};
    }
    int length = 0;
    int location = 0;
    for (int i = 0; i != 32; ++i) {
      int current_run = 0;
      int location_of_current_run = i;
      while (i != 32 && (tagged_index & (1 << i)) == 0) {
        ++current_run;
        ++i;
      }
      if (current_run != 0) {
        --i;
      }
      if (current_run > length) {
        length = current_run;
        location = location_of_current_run;
      }
    }
    if (length < 8) {
      return {-1, tagged_index};
    }
    auto index_mask = llvm::maskTrailingOnes<uint32_t>(location);
    auto ir_id = (llvm::reverseBits(tagged_index & ~index_mask) >> 2) - 1;
    auto index = tagged_index & index_mask;
    return {ir_id, index};
  }
  auto Remove(int32_t tagged_index) const -> int32_t {
    if ((llvm::reverseBits(2) & tagged_index) == 0) {
      CARBON_DCHECK(tagged_index < initial_reserved_ids_,
                    "This untagged index is outside the initial reserved ids "
                    "and should have been tagged.");
      return tagged_index;
    }
    auto index = tagged_index ^ id_tag_;
    CARBON_DCHECK(index >= initial_reserved_ids_,
                  "When removing tagging bits, found an index that "
                  "shouldn't've been tagged in the first place.");
    return index;
  }
  int32_t id_tag_;
  int32_t initial_reserved_ids_;
};

// A simple wrapper for accumulating values, providing IDs to later retrieve the
// value. This does not do deduplication.
template <typename IdT, typename ValueT>
class ValueStore
    : public std::conditional<std::is_base_of_v<Printable<ValueT>, ValueT>,
                              Yaml::Printable<ValueStore<IdT, ValueT>>,
                              Internal::ValueStoreNotPrintable> {
 public:
  using IdType = IdT;
  using ValueType = ValueStoreTypes<ValueT>::ValueType;
  using RefType = ValueStoreTypes<ValueT>::RefType;
  using ConstRefType = ValueStoreTypes<ValueT>::ConstRefType;

  // A range over references to the values in a ValueStore, returned from
  // `ValueStore::values()`. Hides the complex type name of the iterator
  // internally to provide a type name (`Range`) that can be
  // referred to without auto and templates.
  class Range {
   public:
    explicit Range(const ValueStore& store [[clang::lifetimebound]])
        : flattened_range_(MakeFlattenedRange(store)) {}

    auto begin() const -> auto { return flattened_range_.begin(); }
    auto end() const -> auto { return flattened_range_.end(); }

   private:
    // Flattens the range of `Chunk`s of `ValueType`s into a single
    // range of `ValueType`s.
    static auto MakeFlattenedRange(const ValueStore& store) -> auto {
      // Because indices into `ValueStore` are all sequential values from 0, we
      // can use llvm::seq to walk all indices in the store.
      return llvm::map_range(
          llvm::seq(store.size_),
          [&](int32_t i) -> ConstRefType { return store.Get(IdType(i)); });
    }

    using FlattenedRangeType =
        decltype(MakeFlattenedRange(std::declval<const ValueStore&>()));
    FlattenedRangeType flattened_range_;
  };

  ValueStore() = default;
  explicit ValueStore(IdTag tag) : tag_(tag) {}

  // Stores the value and returns an ID to reference it.
  auto Add(ValueType value) -> IdType {
    // This routine is especially hot and the check here relatively expensive
    // for the value provided, so only do this in non-optimized builds to make
    // tracking down issues easier.
    CARBON_DCHECK(size_ < std::numeric_limits<int32_t>::max(), "Id overflow");

    IdType id(tag_.Apply(size_));
    auto [chunk_index, pos] = RawIndexToChunkIndices(size_);
    ++size_;

    CARBON_DCHECK(static_cast<size_t>(chunk_index) <= chunks_.size(),
                  "{0} <= {1}", chunk_index, chunks_.size());
    if (static_cast<size_t>(chunk_index) == chunks_.size()) {
      chunks_.emplace_back();
    }

    CARBON_DCHECK(pos == chunks_[chunk_index].size());
    chunks_[chunk_index].Add(std::move(value));
    return id;
  }

  // Returns a mutable value for an ID.
  auto Get(IdType id) -> RefType {
    auto [chunk_index, pos] = IdToChunkIndices(id);
    return chunks_[chunk_index].Get(pos);
  }

  // Returns the value for an ID.
  auto Get(IdType id) const -> ConstRefType {
    auto [chunk_index, pos] = IdToChunkIndices(id);
    return chunks_[chunk_index].Get(pos);
  }

  // Reserves space.
  auto Reserve(int32_t size) -> void {
    if (size <= size_) {
      return;
    }
    auto [final_chunk_index, _] = RawIndexToChunkIndices(size - 1);
    chunks_.resize(final_chunk_index + 1);
  }

  // Grows the ValueStore to `size`. Fills entries with `default_value`.
  auto Resize(int32_t size, ConstRefType default_value) -> void {
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
      chunks_[begin_chunk_index].UninitializedFill(
          Chunk::Capacity() - begin_pos, default_value);
      for (auto i = begin_chunk_index + 1; i < end_chunk_index; ++i) {
        chunks_[i].UninitializedFill(Chunk::Capacity(), default_value);
      }
      chunks_[end_chunk_index].UninitializedFill(end_pos + 1, default_value);
    }

    // Update size.
    size_ = size;
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
                  Chunk::CapacityBytes * chunks_.size());
  }

  auto size() const -> size_t { return size_; }

  // Makes an iterable range over references to all values in the ValueStore.
  auto values() [[clang::lifetimebound]] -> auto {
    return llvm::map_range(
        llvm::seq(size_), [&](int32_t i) -> RefType { return Get(IdType(i)); });
  }
  auto values() const [[clang::lifetimebound]] -> Range { return Range(*this); }

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
    auto index_to_id = [&](int32_t i) -> const std::pair<IdType, ConstRefType> {
      IdType id(tag_.Apply(i));
      return std::pair<IdType, ConstRefType>(id, Get(id));
    };
    // Because indices into `ValueStore` are all sequential values from 0, we
    // can use llvm::seq to walk all indices in the store.
    return llvm::map_range(llvm::seq(size_), index_to_id);
  }

  auto GetIdTag() const -> IdTag { return tag_; }
  auto GetRawIndex(IdT id) const -> int32_t {
    auto index = tag_.Remove(id.index);
    CARBON_DCHECK(index >= 0, "{0}", index);
    // Attempt to decompose id.index to include extra detail in the check here
#ifndef NDEBUG
    if (index >= size_) {
      auto [ir_id, decomposed_index] = IdTag::DecomposeWithBestEffort(id.index);
      CARBON_DCHECK(
          index < size_,
          "Untagged index was outside of container range. Possibly tagged "
          "index {0}. Best-effort decomposition: CheckIRId: {1}, index: {2}. "
          "The IdTag for this container is: {3}",
          id.index, ir_id, decomposed_index, tag_.GetCheckIRId());
    }
#endif
    return index;
  }

 private:
  // A chunk of `ValueType`s which has a fixed capacity, but variable size.
  // Tracks the size internally for verifying bounds.
  struct Chunk {
   public:
    // The max size of each chunk allocation for `ValueStore`. This is based on
    // TLB page sizes for the target platform.
    //
    // See https://docs.kernel.org/admin-guide/mm/hugetlbpage.html
    //
    // A 4K chunk size outperforms a 1M chunk size on Linux-x64 and MacOS-arm64
    // in benchmarks and when running file_test.
    //
    // Linux-x64: x64 CPUs support 4K and 2M page sizes, but we see 1M is slower
    // than 4K with tcmalloc in opt builds for our tests.
    //
    // Mac-arm64: arm64 CPUs support 4K, 8K, 64K, 256K, 1M, 4M and up. Like for
    // Linux-x64, 4K outperformed 1M. We didn't try other sizes yet.
    //
    // TODO: Is there a more optimize size for Mac-arm64? What should
    // Linux-arm64 and Mac-x64 use? What should Windows use?
    //
    // TODO: The previous SmallVector<ValueType> seems to outperform 4K chunks
    // (they may be slower by up to 5%) in benchmarks. Find ways to make
    // chunking faster. Should successive chunks get larger in size? That will
    // greatly complicate math for choosing a chunk though.
    static constexpr auto MaxAllocationBytes() -> int32_t {
#if !defined(NDEBUG) || LLVM_ADDRESS_SANITIZER_BUILD
      // Use a small size in unoptimized builds to ensure multiple chunks get
      // used. And do the same in ASAN builds to reduce bookkeeping overheads.
      // Using large allocations (e.g. 1M+) incurs a 10x runtime cost for our
      // tests under ASAN.
      return sizeof(ValueType) * 5;
#else
      return 4 * 1024;
#endif
    }

    // The number of elements stored in each chunk allocation.
    //
    // The number must be a power of two so that that there are no unused values
    // in bits indexing into the allocation.
    static constexpr auto Capacity() -> int32_t {
      constexpr auto MaxElements = MaxAllocationBytes() / sizeof(ValueType);
      return std::bit_floor(MaxElements);
    }

    // The number of bits needed to index each element in a chunk allocation.
    static constexpr auto IndexBits() -> int32_t {
      static_assert(Capacity() > 0);
      return std::bit_width(uint32_t{Capacity() - 1});
    }

    static constexpr auto CapacityBytes = Capacity() * sizeof(ValueType);

    explicit Chunk()
        : buf_(reinterpret_cast<ValueType*>(
              llvm::allocate_buffer(CapacityBytes, alignof(ValueType)))) {}

    // Moving leaves nullptr behind in the moved-from object so that the
    // destructor is a no-op (preventing double free).
    Chunk(Chunk&& rhs) noexcept
        : buf_(std::exchange(rhs.buf_, nullptr)), num_(rhs.num_) {}

    auto operator=(Chunk&& rhs) noexcept -> Chunk& {
      buf_ = std::exchange(rhs.buf_, nullptr);
      num_ = rhs.num_;
      return *this;
    }

    ~Chunk() {
      if (buf_) {
        if constexpr (!std::is_trivially_destructible_v<ValueType>) {
          std::destroy_n(buf_, num_);
        }
        llvm::deallocate_buffer(buf_, CapacityBytes, alignof(ValueType));
      }
    }

    auto Get(int32_t i) -> ValueType& {
      CARBON_DCHECK(i < num_, "{0}", i);
      return buf_[i];
    }
    auto Get(int32_t i) const -> const ValueType& {
      CARBON_DCHECK(i < num_, "{0}", i);
      return buf_[i];
    }

    auto Add(ValueType&& value) -> void {
      CARBON_DCHECK(num_ < Capacity());
      std::construct_at(buf_ + num_, std::move(value));
      ++num_;
    }

    // Fills `fill_count` entries with `default_value`, increasing the size
    // respectively.
    auto UninitializedFill(int32_t fill_count, ConstRefType default_value)
        -> void {
      CARBON_DCHECK(num_ + fill_count <= Capacity());
      std::uninitialized_fill_n(buf_ + num_, fill_count, default_value);
      num_ += fill_count;
    }

    auto size() const -> int32_t { return num_; }

   private:
    // Verify using an `int32_t` for `num_` is sound.
    static_assert(Capacity() <= std::numeric_limits<int32_t>::max());

    ValueType* buf_;
    int32_t num_ = 0;
  };

  // Converts a raw index into an index into the set of chunks, and an offset
  // into that specific chunk. Looks for index overflow in non-optimized builds.
  static auto RawIndexToChunkIndices(int32_t index)
      -> std::pair<int32_t, int32_t> {
    constexpr auto LowBits = Chunk::IndexBits();

    // Verify there are no unused bits when indexing up to the `Capacity`. This
    // ensures that ids are contiguous values from 0, as if the values were all
    // stored in a single array, and allows using the ids to index into other
    // arrays.
    static_assert((1 << LowBits) == Chunk::Capacity());
    // Simple check to make sure nothing went wildly wrong with the `Capacity`,
    // and we have some room for a chunk index, and that shifting by the number
    // of bits won't be UB in an int32_t.
    static_assert(LowBits < 30);

    // The index of the chunk is the high bits.
    auto chunk = index >> LowBits;
    // The index into the chunk is the low bits.
    auto pos = index & ((1 << LowBits) - 1);
    return {chunk, pos};
  }

  // Converts an id into an index into the set of chunks, and an offset into
  // that specific chunk.
  auto IdToChunkIndices(IdType id) const -> std::pair<int32_t, int32_t> {
    return RawIndexToChunkIndices(GetRawIndex(id));
  }

  // Number of elements added to the store. The number should never exceed what
  // fits in an `int32_t`, which is checked in non-optimized builds in Add().
  int32_t size_ = 0;

  IdTag tag_;

  // Storage for the `ValueType` objects, indexed by the id. We use a vector of
  // chunks of `ValueType` instead of just a vector of `ValueType` so that
  // addresses of `ValueType` objects are stable. This allows the rest of the
  // toolchain to hold references into `ValueStore` without having to worry
  // about invalidation and use-after-free. We ensure at least one Chunk is held
  // inline so that in the case where there is only a single Chunk (i.e. small
  // files) we can avoid one indirection.
  llvm::SmallVector<Chunk, 1> chunks_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_VALUE_STORE_H_
