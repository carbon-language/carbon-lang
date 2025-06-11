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

// Ids which are stored in a `ValueStore` have a `ValueType` which indicates the
// type of value held in the `ValueStore`.
template <class IdT>
concept IdHasValueType = requires { typename IdT::ValueType; };

// The max size of each chunk allocation for `ValueStore`. This is based on TLB
// page sizes for the target platform.
//
// See https://docs.kernel.org/admin-guide/mm/hugetlbpage.html
//
// A 4K chunk size outperforms a 1M chunk size on Linux-x64 and MacOS-arm64 in
// benchmarks and when running file_test.
//
// Linux-x64: x64 CPUs support 4K and 2M page sizes, but we see 1M is slower
// than 4K with tcmalloc in opt builds for our tests.
//
// Mac-arm64: arm64 CPUs support 4K, 8K, 64K, 256K, 1M, 4M and up. Like for
// Linux-x64, 4K outperformed 1M. We didn't try other sizes yet.
//
// TODO: Is there a more optimize size for Mac-arm64? What should Linux-arm64
// and Mac-x64 use? What should Windows use?
//
// TODO: The previous SmallVector<ValueType> seems to outperform 4K chunks (they
// may be slower by up to 5%) in benchmarks. Find ways to make chunking faster.
// Should successive chunks get larger in size? That will greatly complicate
// math for choosing a chunk though.
template <class IdT>
  requires(IdHasValueType<IdT>)
static constexpr auto PlatformChunkMaxAllocationBytes() -> int32_t {
#if !defined(NDEBUG) || LLVM_ADDRESS_SANITIZER_BUILD
  // Use a small size in unoptimized builds to ensure multiple chunks get used.
  // And do the same in ASAN builds to reduce bookkeeping overheads. Using large
  // allocations (e.g. 1M+) incurs a 10x runtime cost for our tests under ASAN.
  return sizeof(typename IdT::ValueType) * 5;
#else
  return 4 * 1024;
#endif
}

// The number of elements stored in each chunk allocation.
//
// The number must be a power of two so that that there are no unused values in
// bits indexing into the allocation.
template <class IdT>
  requires(IdHasValueType<IdT>)
constexpr auto PlatformChunkCapacity() -> int32_t {
  constexpr auto MaxElements =
      PlatformChunkMaxAllocationBytes<IdT>() / sizeof(typename IdT::ValueType);
  return std::bit_floor(MaxElements);
}

// The number of bits needed to index each element in a chunk allocation.
template <class IdT>
  requires(IdHasValueType<IdT>)
constexpr auto PlatformChunkIndexBits() -> int32_t {
  static_assert(PlatformChunkCapacity<IdT>() > 0);
  return std::bit_width(uint32_t{PlatformChunkCapacity<IdT>() - 1});
}

// Converts an id into an index into the set of chunks, and an offset into that
// specific chunk. Looks for index overflow in non-optimized builds.
template <typename IdT>
  requires(IdHasValueType<IdT>)
inline auto IdToChunkIndices(IdT id) -> std::pair<int32_t, int32_t> {
  constexpr auto LowBits = PlatformChunkIndexBits<IdT>();

  // Verify there are no unused bits when indexing up to the
  // PlatformChunkCapacity(). This ensures that ids are contiguous values
  // from 0, as if the values were all stored in a single array, and allows
  // using the ids to index into other arrays.
  static_assert((1 << LowBits) == PlatformChunkCapacity<IdT>());
  // Simple check to make sure nothing went wildly wrong with the
  // PlatformChunkCapacity, and we have some room for a chunk index, and
  // that shifting by the number of bits won't be UB in an int32_t.
  static_assert(LowBits < 30);

  // The index of the chunk is the high bits.
  auto chunk = id.index >> LowBits;
  // The index into the chunk is the low bits.
  auto pos = id.index & ((1 << LowBits) - 1);
  return {chunk, pos};
}

// A chunk of `ValueType`s which has a fixed capacity, but variable size. Tracks
// the size internally for verifying bounds.
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
  // Verify using an `int32_t` for `num_` is sound.
  static_assert(Capacity <= std::numeric_limits<int32_t>::max());

  ValueType* buf_;
  int32_t num_ = 0;
};

}  // namespace Carbon::Internal

#endif  // CARBON_TOOLCHAIN_BASE_VALUE_STORE_CHUNK_H_
