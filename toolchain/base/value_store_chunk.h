// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_VALUE_STORE_CHUNK_H_
#define CARBON_TOOLCHAIN_BASE_VALUE_STORE_CHUNK_H_

#include <bit>
#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include "common/check.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemAlloc.h"
#include "toolchain/base/mem_usage.h"

namespace Carbon::Internal {

// Ids which are stored in a ValueStore have a ValueType which indicates the
// type of value held in the ValueStore.
template <class IdT>
concept IdHasValueType = requires { typename IdT::ValueType; };

// The max size of each chunk allocation for ValueStore. This is based on TLB
// page sizes for the target platform.
//
// See https://docs.kernel.org/admin-guide/mm/hugetlbpage.html
template <class IdT>
  requires(IdHasValueType<IdT>)
static constexpr auto PlatformChunkMaxAllocationBytes() -> int {
#if (!defined(NDEBUG) || LLVM_ADDRESS_SANITIZER_BUILD)
  // Use a small size in unoptimized builds to ensure multiple chunks get used.
  // And do the same in ASAN builds to reduce bookkeeping overheads. Using large
  // allocations (e.g. 1M+) incurs a 10x runtime cost for our tests under ASAN.
  return sizeof(typename IdT::ValueType) * 5;
#else
  // TODO: Should ia64 use 1M or 4M? Should Windows and Mac use different sizes?

  // x64 CPUs support 4K and 2M page sizes, but we see 1M is slower than 1K with
  // tcmalloc in opt builds for our tests.
  //
  // FIXME: More benchmarking needed.
  return 4 * 1024;
#endif
}

// The number of elements stored in each chunk allocation.
//
// The number must be a power of two so that that there are no unused values in
// bits indexing into the allocation.
template <class IdT>
  requires(IdHasValueType<IdT>)
static constexpr auto PlatformChunkCapacity() -> int {
  constexpr auto MaxElements =
      PlatformChunkMaxAllocationBytes<IdT>() / sizeof(typename IdT::ValueType);
  return std::bit_floor(MaxElements);
}

// The number of bits needed to index each element in a chunk allocation.
template <class IdT>
  requires(IdHasValueType<IdT>)
static constexpr auto PlatformChunkCapacityBits() -> int {
  static_assert(PlatformChunkCapacity<IdT>() > 1);
  int bits = 0;
  for (auto size = PlatformChunkCapacity<IdT>(); size > 1; size /= 2) {
    ++bits;
  }
  return bits;
}

// Converts an id into an index into the set of chunks, and an offset into that
// specific chunk.
template <typename IdT>
  requires(IdHasValueType<IdT>)
static constexpr auto IdToChunkIndices(IdT id) -> std::pair<int, int> {
  constexpr auto LowBits = PlatformChunkCapacityBits<IdT>();

  // Verify there are no unused bits when indexing up to the
  // PlatformChunkCapacity(). This ensures that ids are contiguous values
  // from 0, as if the values were all stored in a single array, and allows
  // using the ids to index into other arrays.
  static_assert((1 << LowBits) == PlatformChunkCapacity<IdT>());
  // Simple check to make sure nothing went wildly wrong with the
  // PlatformChunkCapacity, and we have some room for a chunk index, and
  // that shifting by the number of bits won't be UB in an int32_t.
  static_assert(LowBits < 30);

  return {
      id.index >> LowBits,
      id.index & ((1 << LowBits) - 1),
  };
}

// Converts an index into the set of chunks, and an offset into that specific
// chunk, into an id.
template <typename IdT>
  requires(IdHasValueType<IdT>)
static constexpr auto ChunkIndicesToId(int chunk, int pos) -> IdT {
  constexpr auto LowBits = PlatformChunkCapacityBits<IdT>();
  // We can only use 31 bits in total because the sign bit is used for making
  // negative ids, which are not found in the ValueStore.
  constexpr auto HighBits = 31 - LowBits;
  // This routine is especially hot and the check here relatively expensive for
  // the value provided, so only do this in debug builds to make tracking down
  // issues easier.
  CARBON_DCHECK(chunk < (1 << HighBits), "Id overflow (high bits)");
  CARBON_DCHECK(pos < (1 << LowBits), "Id overflow (low bits)");
  return IdT((chunk << LowBits) + pos);
}

// A chunk of `ValueType`s which has a fixed capacity, but variable size, and is
// an iterable range.
template <typename IdT, class ValueType>
struct ValueStoreChunk {
 public:
  static constexpr auto Capacity = Internal::PlatformChunkCapacity<IdT>();
  static constexpr auto CapacityBytes = Capacity * sizeof(ValueType);

  explicit ValueStoreChunk()
      : buf_(reinterpret_cast<ValueType*>(
            llvm::allocate_buffer(CapacityBytes, alignof(ValueType)))) {}

  // Moving leaves nullptr behind in the moved-from object so that the
  // destructor is a no-op (preventing double free).
  ValueStoreChunk(ValueStoreChunk&& rhs) noexcept
      : buf_(std::exchange(rhs.buf_, nullptr)), num_(rhs.num_) {}

  auto operator=(ValueStoreChunk&& rhs) noexcept -> ValueStoreChunk& {
    buf_ = std::exchange(rhs.buf_, nullptr);
    num_ = rhs.num_;
    return *this;
  }

  ~ValueStoreChunk() {
    if (buf_) {
      if constexpr (!std::is_trivially_destructible_v<ValueType>) {
        std::destroy_n(buf_, num_);
      }
      llvm::deallocate_buffer(buf_, CapacityBytes, alignof(ValueType));
    }
  }

  // Allow the chunk to act as a range for being iterated.
  auto begin() const -> const ValueType* {
    CARBON_DCHECK(buf_, "iterating after moved-from");
    return buf_;
  }
  auto end() const -> const ValueType* {
    CARBON_DCHECK(buf_, "iterating after moved-from");
    return buf_ + num_;
  }

  // Verify using an `int32_t` for `num_` is sound.
  static_assert(Capacity <= std::numeric_limits<int32_t>::max());

  auto at(int32_t i) -> ValueType& {
    CARBON_CHECK(i < num_, "{0}", i);
    return buf_[i];
  }
  auto at(int32_t i) const -> const ValueType& {
    CARBON_CHECK(i < num_, "{0}", i);
    return buf_[i];
  }

  auto push(ValueType&& value) -> void {
    CARBON_CHECK(num_ < Capacity);
    std::construct_at(buf_ + num_, std::move(value));
    ++num_;
  }

  auto size() const -> int32_t { return num_; }

 private:
  ValueType* buf_;
  int32_t num_ = 0;
};

}  // namespace Carbon::Internal

#endif  // CARBON_TOOLCHAIN_BASE_VALUE_STORE_CHUNK_H_
