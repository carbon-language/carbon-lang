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

// The max size of each chunk allocation for ValueStore. This is based on TLB
// page sizes for the target platform.
//
// See https://docs.kernel.org/admin-guide/mm/hugetlbpage.html
template <class IdT>
  requires requires { typename IdT::ValueType; }
static constexpr auto PlatformChunkMaxAllocationBytes() -> int {
#if !defined(NDEBUG) || defined(FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION)
  // Use a small size in debug/fuzzer builds to ensure multiple chunks get used.
  return sizeof(typename IdT::ValueType) * 5;
#else
  // TODO: Should ia64 use 1M or 4M? Should Windows and Mac use different sizes?

  // Linux x64 uses 2M, as x64 CPUs support 4K and 2M page sizes.
  return 2 * 1024 * 1024;
#endif
}

// The number of elements stored in each chunk allocation.
//
// The number must be a power of two so that that there are no unused values in
// bits indexing into the allocation.
template <class IdT>
  requires requires { typename IdT::ValueType; }
static constexpr auto PlatformChunkCapacitySize() -> int {
  constexpr auto MaxElements =
      PlatformChunkMaxAllocationBytes<IdT>() / sizeof(typename IdT::ValueType);
  constexpr auto Pow2MaxElements = std::bit_ceil(MaxElements);
  if (Pow2MaxElements > MaxElements) {
    return Pow2MaxElements / 2;
  }
  return Pow2MaxElements;
}

// The number of bits needed to index each element in a chunk allocation.
template <class IdT>
  requires requires { typename IdT::ValueType; }
static constexpr auto PlatformChunkCapacityBits() -> int {
  static_assert(PlatformChunkCapacitySize<IdT>() > 1);
  int bits = 0;
  for (auto size = PlatformChunkCapacitySize<IdT>(); size > 1; size /= 2) {
    ++bits;
  }
  return bits;
}

// Converts an id into an index into the set of chunks, and an offset into that
// specific chunk.
template <typename IdT>
  requires requires { typename IdT::ValueType; }
static constexpr auto IdToChunkIndices(IdT id) -> std::pair<int, int> {
  constexpr auto LowBits = PlatformChunkCapacityBits<IdT>();

  // Verify there are no unused bits when indexing up to the
  // PlatformChunkCapacitySize(). This ensures that ids are contiguous values
  // from 0, as if the values were all stored in a single array, and allows
  // using the ids to index into other arrays.
  static_assert((1 << LowBits) == PlatformChunkCapacitySize<IdT>());
  // Simple check to make sure nothing went wildly wrong with the
  // PlatformChunkCapacitySize, and we have some room for a chunk index, and
  // that shifting by the number of bits won't be UB in an int32_t.
  static_assert(LowBits < 24);

  return {
      id.index >> LowBits,
      id.index & ((1 << LowBits) - 1),
  };
}

// Converts an index into the set of chunks, and an offset into that specific
// chunk, into an id.
template <typename IdT>
  requires requires { typename IdT::ValueType; }
static constexpr auto ChunkIndicesToId(int chunk, int pos) -> IdT {
  constexpr auto LowBits = PlatformChunkCapacityBits<IdT>();
  // We can only use 31 bits in total because the sign bit is used for making
  // negative ids which are not found in the ValueStore.
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
  static constexpr auto Capacity = Internal::PlatformChunkCapacitySize<IdT>();
  static constexpr auto CapacityBytes = Capacity * sizeof(ValueType);

  ValueStoreChunk()
      : buf(reinterpret_cast<ValueType*>(llvm::safe_malloc(CapacityBytes))) {}

  ValueStoreChunk(ValueStoreChunk&& rhs) noexcept
      : buf(std::exchange(rhs.buf, nullptr)), num(rhs.num) {}

  auto operator=(ValueStoreChunk&& rhs) noexcept -> ValueStoreChunk& {
    buf = std::exchange(rhs.buf, nullptr);
    num = rhs.num;
    return *this;
  }

  ~ValueStoreChunk() {
    if (buf) {
      if constexpr (!std::is_trivially_destructible_v<ValueType>) {
        std::destroy_n(buf, num);
      }
      free(buf);
    }
  }

  // Allow the chunk to act as a range for being iterated.
  auto begin() const -> const ValueType* {
    CARBON_DCHECK(buf, "iterating after moved-from");
    return buf;
  }
  auto end() const -> const ValueType* {
    CARBON_DCHECK(buf, "iterating after moved-from");
    return buf + num;
  }

  // Verify using an `int32_t` for `num` is sound.
  static_assert(Capacity <= std::numeric_limits<int32_t>::max());

  auto at(int32_t i) -> ValueType& {
    CARBON_CHECK(i < num, "{0}", i);
    return buf[i];
  }
  auto at(int32_t i) const -> const ValueType& {
    CARBON_CHECK(i < num, "{0}", i);
    return buf[i];
  }

  auto push(ValueType&& value) -> void {
    CARBON_CHECK(num < Capacity);
    std::construct_at(buf + num, std::move(value));
    ++num;
  }

  auto size() const -> int32_t { return num; }

 private:
  ValueType* buf;
  int32_t num = 0;
};

}  // namespace Carbon::Internal

#endif  // CARBON_TOOLCHAIN_BASE_VALUE_STORE_CHUNK_H_
