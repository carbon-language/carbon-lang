// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_STABLE_STACK_H_
#define CARBON_TOOLCHAIN_BASE_STABLE_STACK_H_

#include <algorithm>
#include <bit>
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
// chunks, which makes `operator==` simpler to implement. The stack uses a
// configurable value for the size of the first allocation.
//
// Like a std::vector, the stack does not free memory after Pop(). This means
// the stack retains capacity for holding the maximum number of values it ever
// held.
//
// TODO: The FirstCapacity being in the type allows for some simplifications
// internally where we can assume two stacks of the same type have the same
// chunk sizes. The typical use of StableStack is currently that we have one
// StableStack type for a given `T`, as they all have the same FirstCapacity. If
// we end up with many different StableStack types for the same `T` with
// different FirstCapacity, we should consider moving it to a runtime value.
template <typename T, int32_t FirstCapacity = 32>
class StableStack {
  static_assert(FirstCapacity > 0);
  static_assert(std::has_single_bit(static_cast<uint32_t>(FirstCapacity)),
                "FirstCapacity must be a power of 2");

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
    used_chunks_ = rhs.used_chunks_;
    size_in_last_chunk_ = rhs.size_in_last_chunk_;
    CopyConstructFrom(rhs);
    return *this;
  }

  // Moving the stack. The moved-from `rhs` stack will be left empty.
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

  // Returns a reference to an element in the stack at distance `n` from the
  // back of the stack.
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

  // Pushes a new element onto the back of the stack, constructed in place from
  // the given arguments.
  template <typename... Args>
  auto Emplace(Args&&... args) -> void {
    auto* slot = PushSlot();
    __asan_unpoison_memory_region(slot, sizeof(T));
    std::construct_at<T>(slot, std::forward<Args>(args)...);
  }

  // Removes the most recently pushed element from the back of the stack.
  auto Pop() -> void {
    CARBON_CHECK(!empty(), "Pop on an empty stack");
    auto [chunk, index] = GetIndexAtBack();
    auto& ref = Get(chunk, index);

    ShrinkSize();

    std::destroy_at(&ref);
    __asan_poison_memory_region(&ref, sizeof(T));
  }

  // Removes the most recently pushed element from the back of the stack and
  // returns it.
  auto PopVal() -> T {
    CARBON_CHECK(!empty(), "Pop on an empty stack");
    auto [chunk, index] = GetIndexAtBack();
    auto& ref = Get(chunk, index);

    ShrinkSize();

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
      AppendChunk();
      current += GetChunkCapacity(chunks_.size() - 1);
    }
  }

  // Whether there are any elements pushed onto the stack.
  auto empty() const -> bool { return used_chunks_ == 0; }
  // The number of elements pushed onto the stack.
  auto size() const -> int32_t {
    auto full_chunks_capacity =
        used_chunks_ > 0 ? GetSumCapacity(used_chunks_ - 1) : 0;
    return full_chunks_capacity + size_in_last_chunk_;
  }
  // The number of elements that the stack can hold without requiring further
  // allocation.
  auto capacity() const -> int32_t { return GetSumCapacity(chunks_.size()); }

  // Compares the contents of two stacks for equality.
  friend auto operator==(const StableStack& lhs, const StableStack& rhs) -> bool
    requires std::equality_comparable<T>
  {
    if (lhs.used_chunks_ != rhs.used_chunks_) {
      return false;
    }
    if (lhs.size_in_last_chunk_ != rhs.size_in_last_chunk_) {
      return false;
    }
    for (auto chunk = 0; chunk < lhs.used_chunks_; ++chunk) {
      const T* lhs_array = lhs.chunks_[chunk].array;
      const T* rhs_array = rhs.chunks_[chunk].array;
      int32_t cap = GetChunkCapacity(chunk);
      auto size = chunk < lhs.used_chunks_ - 1 ? cap : lhs.size_in_last_chunk_;
      if (!std::equal(lhs_array, lhs_array + size, rhs_array)) {
        return false;
      }
    }
    return true;
  }

  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Add(label.str(), size() * sizeof(T), capacity() * sizeof(T));
  }

 private:
  // The total capacity of the first `num_chunks` chunks.
  static constexpr auto GetSumCapacity(int32_t num_chunks) -> int32_t {
    return ((1 << num_chunks) - 1) * FirstCapacity;
  }
  // The capacity of an individual chunk at index `chunk`.
  static constexpr auto GetChunkCapacity(int32_t chunk) -> int32_t {
    // We use a growth factor of 2 so that we can do bitshifts in
    // GetSumCapacity() to compute total capacity in O(1) time.
    return FirstCapacity << chunk;
  }
  // Performs the inverse of GetSumCapacity.
  static constexpr auto NumChunksNeededForCapacity(int32_t cap) -> int32_t {
    // Given this formula, we solve for `num_chunks`.
    // cap = ((1 << num_chunks) - 1) * FirstCapacity;
    auto ucap = static_cast<uint32_t>(cap);
    // Shifting by this simulates division by FirstCapacity.
    auto division = std::bit_width(static_cast<uint32_t>(FirstCapacity - 1));
    return std::bit_width((ucap + (FirstCapacity - 1)) >> division);
  }

  struct Chunk {
    T* array [[clang::require_explicit_initialization]];
  };

  // Copy-constructs elements into this stack from `rhs`.
  auto CopyConstructFrom(const StableStack& rhs) -> void {
    if (rhs.empty()) {
      return;
    }

    chunks_.clear();
    chunks_.reserve(used_chunks_);
    for (auto chunk = 0; chunk < used_chunks_; ++chunk) {
      T* rhs_array = rhs.chunks_[chunk].array;
      auto cap = GetChunkCapacity(chunk);
      T* array = std::allocator<T>().allocate(cap);
      chunks_.push_back({.array = array});
      auto size = chunk < used_chunks_ - 1 ? cap : size_in_last_chunk_;
      std::uninitialized_copy_n(rhs_array, size, array);
      if (size < cap) {
        __asan_poison_memory_region(array + size, (cap - size) * sizeof(T));
      }
    }
  }

  // Destroy all the elements in the stack and free the arrays in the chunks.
  // Leaves the stack in an empty state.
  auto DestroyAll() -> void {
    // Make `this` appear empty while destroying its elements in case they try
    // to use the stack mid-destruction.
    auto size_in_last_chunk = std::exchange(size_in_last_chunk_, 0);
    auto used_chunks = std::exchange(used_chunks_, 0);
    auto chunks = std::exchange(chunks_, {});

    for (auto chunk = 0; chunk < static_cast<int32_t>(chunks.size()); ++chunk) {
      auto* array = chunks[chunk].array;
      auto cap = GetChunkCapacity(chunk);
      if (chunk < used_chunks - 1) {
        std::destroy_n(array, cap);
      } else if (chunk == used_chunks - 1) {
        std::destroy_n(array, size_in_last_chunk);
      }
      std::allocator<T>().deallocate(array, cap);
    }

    CARBON_CHECK(chunks_.empty(),
                 "Mutation of StableStack while destroying its elements");
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
    // exactly `i + 1` elements, so that `i` is the last element. Then the last
    // element in that stack is the index we want to compute.
    auto capacity_to_hold_index_i = i + 1;
    auto num_chunks = NumChunksNeededForCapacity(capacity_to_hold_index_i);
    auto capacity_before_chunk = GetSumCapacity(num_chunks - 1);
    return {num_chunks - 1, i - capacity_before_chunk};
  }

  // Get the element in the specified `chunk` at position `index`.
  auto Get(int32_t chunk, int32_t index) -> T& {
    CARBON_DCHECK(chunk >= 0);
    CARBON_DCHECK(chunk < used_chunks_);
    CARBON_DCHECK(index >= 0);
    auto cap = GetChunkCapacity(chunk);
    auto size = chunk == used_chunks_ - 1 ? size_in_last_chunk_ : cap;
    CARBON_DCHECK(index < size,
                  "Getting index {0} past end of last chunk with size {1}",
                  index, size);
    return chunks_[chunk].array[index];
  }
  auto Get(int32_t chunk, int32_t index) const -> const T& {
    return const_cast<StableStack*>(this)->Get(chunk, index);
  }

  // Allocate memory as needed to get a pointer that we can construct `T` into.
  // The pointer returned is still ASAN poisoned.
  auto PushSlot() -> T* {
    // TODO: It would be faster to make the writes to the returned slot not be
    // dependent on the branches, as then the branches can be done
    // asynchronously by the processor. To do that, we would need to always have
    // the allocation ready for the *next* push. That means eagerly allocating
    // the first chunk at construction time. And it means we can always grab a
    // pointer here immediately. Then the rest of the branches and work in here
    // does not affect the pointer, nor the writes through the pointer. For a
    // moved-from stack we don't want to have to allocate in the RHS, so we can
    // say we leave moved-from stacks with an empty set of chunks, and DCHECK
    // against use-after-move.

    if (used_chunks_ > 0 &&
        size_in_last_chunk_ < GetChunkCapacity(used_chunks_ - 1)) {
      // Add a slot in the last used chunk.
      T* slot = chunks_[used_chunks_ - 1].array + size_in_last_chunk_;
      ++size_in_last_chunk_;
      return slot;
    }

    if (used_chunks_ < static_cast<int32_t>(chunks_.size())) {
      // Add a slot in an unused chunk that already exists.
      T* slot = chunks_[used_chunks_].array;
      ++used_chunks_;
      size_in_last_chunk_ = 1;
      return slot;
    }

    // Add a slot in a new chunk.
    auto& new_chunk = AppendChunk();
    ++used_chunks_;
    size_in_last_chunk_ = 1;
    return new_chunk.array;
  }
  auto AppendChunk() -> Chunk& {
    auto next_cap = GetChunkCapacity(chunks_.size());
    T* array = std::allocator<T>().allocate(next_cap);
    __asan_poison_memory_region(array, next_cap * sizeof(T));
    chunks_.push_back({.array = array});
    return chunks_.back();
  }

  // Shrinks the recorded size of the stack by one. Does not actually destroy
  // anything.
  auto ShrinkSize() -> void {
    size_in_last_chunk_--;
    if (size_in_last_chunk_ == 0) {
      used_chunks_--;
      if (used_chunks_ > 0) {
        size_in_last_chunk_ = GetChunkCapacity(used_chunks_ - 1);
      } else {
        // Stack is empty now.
      }
    }
  }

  llvm::SmallVector<Chunk> chunks_;
  // The number of chunks that are being used in `chunks_`.
  int32_t used_chunks_ = 0;
  // The number of values in the last used chunk.
  int32_t size_in_last_chunk_ = 0;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_STABLE_STACK_H_
