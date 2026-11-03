// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_STABLE_STACK_H_
#define CARBON_TOOLCHAIN_BASE_STABLE_STACK_H_

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include "common/check.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "toolchain/base/mem_usage.h"

namespace Carbon {

// A stack that never reallocates when growing so as to not invalidate pointers
// to objects that remain in the stack.
//
// As a stack, all insertions must be pushed onto the back of the stack, and
// doing so is an O(1) operation. Popping the back of the stack and peeking into
// the stack are also O(1) operations.
//
// The stack is implemented as an array of pointers to arrays. Each successive
// array gains a larger capacity than the last by multiplicative factor, to
// limit the number of allocations to be logarithmic in the number of
// insertions. But in all stacks, we use the same size progression for the
// chunks, which makes `operator==` simpler to implement.
//
// Like a std::vector, the stack does not free memory after Pop(). This means
// the stack retains capacity for holding the maximum number of values it ever
// held.
template <typename T>
class StableStack {
 public:
  StableStack() = default;
  ~StableStack() { DestroyAll(); }

  // Copying the stack.
  StableStack(const StableStack& rhs)
    requires(std::is_copy_constructible_v<T>)
      : used_chunks_(rhs.used_chunks_),
        size_in_last_chunk_(rhs.size_in_last_chunk_) {
    CopyConstructFrom(rhs);
  }
  auto operator=(const StableStack& rhs) -> StableStack&
    requires(std::is_copy_constructible_v<T> && std::is_copy_assignable_v<T>)
  {
    if (this == &rhs) {
      return *this;
    }

    // TODO: We could reuse existing elements in `this` by calling `operator=`
    // on them instead of destroying and reconstructing always.
    //
    // TODO: This is broken for assignment-from-subobject, where we would need
    // to construct all of the new values before destroying the old.
    DestroyAll();
    chunks_.clear();
    used_chunks_ = rhs.used_chunks_;
    size_in_last_chunk_ = rhs.size_in_last_chunk_;
    CopyConstructFrom(rhs);
    return *this;
  }

  // Moving the stack. The moved-from `rhs` stack will be left empty with no
  // capacity.
  StableStack(StableStack&& rhs) noexcept
      : chunks_(std::exchange(rhs.chunks_, {})),
        used_chunks_(std::exchange(rhs.used_chunks_, 0)),
        size_in_last_chunk_(std::exchange(rhs.size_in_last_chunk_, 0)) {}
  auto operator=(StableStack&& rhs) noexcept -> StableStack& {
    if (this == &rhs) {
      return *this;
    }

    // TODO: This is broken for assignment-from-subobject, where this would
    // destroy the `rhs`.
    DestroyAll();
    chunks_ = std::exchange(rhs.chunks_, {});
    used_chunks_ = std::exchange(rhs.used_chunks_, 0);
    size_in_last_chunk_ = std::exchange(rhs.size_in_last_chunk_, 0);
    return *this;
  }

  // Returns a reference to the most recently pushed element at the back the
  // stack.
  auto Peek() const -> const T& {
    CARBON_CHECK(!empty(), "Peek on an empty stack");
    auto [chunk, index] = GetIndexAtBack();
    return Get(chunk, index);
  }
  auto Peek() -> T& {
    CARBON_CHECK(!empty(), "Peek on an empty stack");
    auto [chunk, index] = GetIndexAtBack();
    return Get(chunk, index);
  }

  // Returns a reference to an element in the stack. By default it returns the
  // most recently pushed element at the back of the stack. But `n` may be used
  // to specify a distance from that element into the stack.
  auto Peek(int32_t n) const -> const T& {
    CARBON_CHECK(!empty(), "Peek on an empty stack");
    CARBON_CHECK(n >= 0, "Peek index invalid");
    auto i = size() - 1 - n;
    CARBON_CHECK(i >= 0, "Peek before the start of the stack");
    auto [chunk, index] = GetIndexFromFront(i);
    return Get(chunk, index);
  }
  auto Peek(int32_t n) -> T& {
    CARBON_CHECK(!empty(), "Peek on an empty stack");
    CARBON_CHECK(n >= 0, "Peek index invalid");
    auto i = size() - 1 - n;
    CARBON_CHECK(i >= 0, "Peek before the start of the stack");
    auto [chunk, index] = GetIndexFromFront(i);
    return Get(chunk, index);
  }

  // Pushes a new element onto the back of the stack.
  auto Push(const T& t) -> void {
    auto* slot = PushSlot();
    __asan_unpoison_memory_region(slot, sizeof(T));
    std::construct_at<T>(slot, t);
  }
  auto Push(T&& t) -> void {
    auto* slot = PushSlot();
    __asan_unpoison_memory_region(slot, sizeof(T));
    std::construct_at<T>(slot, std::move(t));
  }

  // Removes the most recently pushed element from the back of the stack and
  // returns it.
  auto Pop() -> T {
    CARBON_CHECK(!empty(), "Pop on an empty stack");
    auto [chunk, index] = GetIndexAtBack();
    auto& ref = Get(chunk, index);

    size_in_last_chunk_--;
    if (size_in_last_chunk_ == 0) {
      used_chunks_--;
      if (used_chunks_ > 0) {
        size_in_last_chunk_ = chunks_[used_chunks_ - 1].capacity;
      }
    }

    auto cleanup = llvm::scope_exit([&] {
      std::destroy_at(&ref);
      __asan_poison_memory_region(&ref, sizeof(T));
    });
    return std::move(ref);
  }

  // Ensure there is memory allocated to hold at least `reserve` elements.
  auto Reserve(int32_t reserve) -> void {
    auto current = capacity();
    while (current < reserve) {
      auto capacity = NextChunkCapacity();
      T* array = std::allocator<T>().allocate(capacity);
      __asan_poison_memory_region(array, capacity * sizeof(T));
      chunks_.push_back({.array = array, .capacity = capacity});
      current += capacity;
    }
  }

  // Whether there are any elements pushed onto the stack.
  auto empty() const -> bool { return used_chunks_ == 0; }
  // The number of elements pushed onto the stack.
  auto size() const -> int32_t {
    if (used_chunks_ == 0) {
      return 0;
    }
    auto full_chunks = used_chunks_ - 1;
    auto full_chunks_capacity = GetSumCapacity(full_chunks);
    return full_chunks_capacity + size_in_last_chunk_;
  }
  // The number of elements that the stack can hold without requiring further
  // allocation.
  auto capacity() const -> int32_t { return GetSumCapacity(chunks_.size()); }

  // Compares the contents of two stacks for equality.
  friend auto operator==(const StableStack& lhs, const StableStack& rhs) -> bool
    requires std::equality_comparable<T>
  {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    if (lhs.empty()) {
      return true;
    }
    CARBON_CHECK(lhs.used_chunks_ == rhs.used_chunks_);
    for (auto chunk = 0; chunk < lhs.used_chunks_ - 1; ++chunk) {
      const auto& l = lhs.chunks_[chunk];
      const auto& r = rhs.chunks_[chunk];
      CARBON_CHECK(l.capacity == r.capacity);
      for (auto index = 0; index < l.capacity; ++index) {
        if (!(lhs.Get(chunk, index) == rhs.Get(chunk, index))) {
          return false;
        }
      }
    }
    CARBON_CHECK(lhs.size_in_last_chunk_ == rhs.size_in_last_chunk_);
    {
      auto chunk = lhs.used_chunks_ - 1;
      const auto& l = lhs.chunks_[chunk];
      const auto& r = rhs.chunks_[chunk];
      CARBON_CHECK(l.capacity == r.capacity);
      for (auto index = 0; index < lhs.size_in_last_chunk_; ++index) {
        if (!(lhs.Get(chunk, index) == rhs.Get(chunk, index))) {
          return false;
        }
      }
    }
    return true;
  }

  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Add(label.str(), size() * sizeof(T), capacity() * sizeof(T));
  }

 private:
  struct Chunk {
    T* array [[clang::require_explicit_initialization]];
    int32_t capacity [[clang::require_explicit_initialization]];
  };

  // Copy-constructs elements into this stack from `rhs`.
  auto CopyConstructFrom(const StableStack& rhs) -> void {
    if (rhs.empty()) {
      return;
    }

    chunks_.reserve(used_chunks_);
    for (int chunk = 0; chunk < used_chunks_ - 1; ++chunk) {
      auto capacity = rhs.chunks_[chunk].capacity;
      T* array = std::allocator<T>().allocate(capacity);
      for (auto index = 0; index < capacity; ++index) {
        T* slot = array + index;
        std::construct_at(slot, rhs.Get(chunk, index));
      }
      chunks_.push_back({.array = array, .capacity = capacity});
    }
    {
      auto chunk = used_chunks_ - 1;
      auto capacity = rhs.chunks_[chunk].capacity;
      T* array = std::allocator<T>().allocate(capacity);
      __asan_poison_memory_region(array + size_in_last_chunk_,
                                  (capacity - size_in_last_chunk_) * sizeof(T));
      for (auto index = 0; index < size_in_last_chunk_; ++index) {
        T* slot = array + index;
        std::construct_at(slot, rhs.Get(chunk, index));
      }
      chunks_.push_back({.array = array, .capacity = capacity});
    }
  }

  // Destroy all the elements in the stack and free the arrays in the chunks.
  // Leaves the stack in an empty state.
  auto DestroyAll() -> void {
    // Make `this` appear empty while destroying its elements in case they try
    // to use the stack mid-destruction.
    auto chunks = std::exchange(chunks_, {});
    auto size_in_last_chunk = std::exchange(size_in_last_chunk_, 0);
    auto used_chunks = std::exchange(used_chunks_, 0);

    for (auto i = 0; i < used_chunks - 1; ++i) {
      auto* array = chunks[i].array;
      auto capacity = chunks[i].capacity;
      std::destroy_n(array, capacity);
      std::allocator<T>().deallocate(array, capacity);
    }
    if (used_chunks) {
      auto i = used_chunks - 1;
      auto* array = chunks[i].array;
      auto capacity = chunks[i].capacity;
      std::destroy_n(array, size_in_last_chunk);
      std::allocator<T>().deallocate(array, capacity);
    }
    for (auto i = used_chunks; i < static_cast<int32_t>(chunks.size()); ++i) {
      auto* array = chunks[i].array;
      auto capacity = chunks[i].capacity;
      std::allocator<T>().deallocate(array, capacity);
    }

    CARBON_CHECK(empty(),
                 "Insertion into StableStack while destroying its elements");
  }

  // Get the index of the element at the back of the stack. This is an O(1)
  // operation.
  auto GetIndexAtBack() const
      -> std::pair<int32_t /*chunk*/, int32_t /*index*/> {
    int32_t chunk = used_chunks_ - 1;
    int32_t index = size_in_last_chunk_ - 1;
    return {chunk, index};
  }

  // Get the index of the element at position `i` from the front of the stack.
  auto GetIndexFromFront(int32_t i) const
      -> std::pair<int32_t /*chunk*/, int32_t /*index*/> {
    // Given a capacity, we can tell how many chunks would be needed. So we
    // compute the number of chunks needed to hold a stack with capacity for
    // exactly `i` elements. Then the last element in that stack is the index we
    // want to compute.
    auto capacity_to_hold_i = i + 1;
    auto num_chunks = NumChunksNeededForCapacity(capacity_to_hold_i);
    auto capacity_before_chunk = GetSumCapacity(num_chunks - 1);
    return {num_chunks - 1, i - capacity_before_chunk};
  }

  // Get the element in the specified `chunk` at position `index`.
  auto Get(int32_t chunk, int32_t index) const -> T& {
    CARBON_DCHECK(chunk >= 0);
    CARBON_DCHECK(chunk < used_chunks_);
    CARBON_DCHECK(index >= 0);
    if (chunk == used_chunks_ - 1) {
      CARBON_DCHECK(index < size_in_last_chunk_);
    } else {
      CARBON_DCHECK(index < chunks_[chunk].capacity);
    }
    return chunks_[chunk].array[index];
  }

  // Allocate memory as needed to get a pointer that we can construct `T` into.
  // The pointer returned is still ASAN poisoned.
  auto PushSlot() -> T* {
    if (empty()) {
      auto& new_chunk = AddChunk();
      size_in_last_chunk_ = 1;
      return new_chunk.array;
    }
    int32_t chunk = used_chunks_ - 1;
    if (size_in_last_chunk_ == chunks_[chunk].capacity) {
      auto& new_chunk = AddChunk();
      size_in_last_chunk_ = 1;
      return new_chunk.array;
    } else {
      int32_t index = size_in_last_chunk_;
      size_in_last_chunk_++;
      return chunks_[chunk].array + index;
    }
  }

  // Append a new chunk to the vector of chunks. Each new chunk has a larger
  // capacity than the last. The chunk is initially fully ASAN poisoned.
  auto AddChunk() -> Chunk& {
    ++used_chunks_;
    if (used_chunks_ > static_cast<int32_t>(chunks_.size())) {
      auto capacity = NextChunkCapacity();
      T* array = std::allocator<T>().allocate(capacity);
      __asan_poison_memory_region(array, capacity * sizeof(T));
      chunks_.push_back({.array = array, .capacity = capacity});
    }
    return chunks_[used_chunks_ - 1];
  }

  // The capacity of the first chunk.
  static auto FirstChunkCapacity() -> int32_t {
    return 2;  // FIXME.
  }
  // The total capacity of the first `num_chunks` chunks.
  static auto GetSumCapacity(int32_t num_chunks) -> int32_t {
    return ((1 << num_chunks) - 1) * FirstChunkCapacity();
  }
  // Performs the inverse of GetSumCapacity.
  static auto NumChunksNeededForCapacity(int32_t cap) -> int32_t {
    // Given this formula, we solve for `num_chunks`.
    // cap = ((1 << num_chunks) - 1) * FirstChunkCapacity();
    auto ucap = static_cast<uint32_t>(cap);
    return std::bit_width(
        ((ucap + (FirstChunkCapacity() - 1)) / FirstChunkCapacity()));
  }
  // The individual capacity to use for the next chunk.
  auto NextChunkCapacity() -> int32_t {
    if (chunks_.empty()) {
      // The size of the first chunk. We use a fairly small number since
      // StableStack is intended for holding larger classes, since those are
      // cases we need to hold references into the stack.
      // TODO: Maybe choose something inversely proportional to sizeof(T)?
      return FirstChunkCapacity();
    }
    // We use a growth factor of 2 so that we can do bitshifts in
    // GetSumCapacity() to compute total capacity in O(1) time.
    return chunks_.back().capacity * 2;
  }

  llvm::SmallVector<Chunk, 1> chunks_;
  // The number of elements in `chunks_` that are being used, which have at
  // least one value stored in them.
  int32_t used_chunks_ = 0;
  // The number of values in the last used chunk in `chunks_`.
  int32_t size_in_last_chunk_ = 0;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_STABLE_STACK_H_
