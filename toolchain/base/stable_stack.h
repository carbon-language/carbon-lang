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
    heap_.clear();
    used_chunks_ = rhs.used_chunks_;
    size_in_last_chunk_ = rhs.size_in_last_chunk_;
    CopyConstructFrom(rhs);
    return *this;
  }

  // Moving the stack. The moved-from `rhs` stack will be left empty with no
  // capacity.
  StableStack(StableStack&& rhs) noexcept
      : heap_(std::exchange(rhs.heap_, {})),
        used_chunks_(std::exchange(rhs.used_chunks_, 0)),
        size_in_last_chunk_(std::exchange(rhs.size_in_last_chunk_, 0)) {
    auto inline_size =
        used_chunks_ > 1 ? FirstChunkCapacity() : size_in_last_chunk_;
    __asan_unpoison_memory_region(inline_.array(), inline_size * sizeof(T));
    std::uninitialized_move_n(rhs.inline_.array(), inline_size,
                              inline_.array());
    std::destroy_n(rhs.inline_.array(), inline_size);
    __asan_poison_memory_region(rhs.inline_.array(),
                                FirstChunkCapacity() * sizeof(T));
  }
  auto operator=(StableStack&& rhs) noexcept -> StableStack& {
    if (this == &rhs) {
      return *this;
    }

    // TODO: This is broken for assignment-from-subobject, where this would
    // destroy the `rhs`.
    DestroyAll();

    // TODO: We could move-assign to existing elements in `inline_` instead of
    // destroy and reconstructing.
    auto inline_size =
        rhs.used_chunks_ > 1 ? FirstChunkCapacity() : rhs.size_in_last_chunk_;
    __asan_unpoison_memory_region(inline_.array(), inline_size * sizeof(T));
    std::uninitialized_move_n(rhs.inline_.array(), inline_size,
                              inline_.array());
    std::destroy_n(rhs.inline_.array(), inline_size);
    __asan_poison_memory_region(rhs.inline_.array(),
                                FirstChunkCapacity() * sizeof(T));

    heap_ = std::exchange(rhs.heap_, {});
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
      if (used_chunks_ > 1) {
        // The last chunk is a heap chunk.
        auto used_heap_chunks = used_chunks_ - 1;
        size_in_last_chunk_ = heap_[used_heap_chunks - 1].capacity;
      } else if (used_chunks_ == 1) {
        // The last chunk is the inline chunk.
        size_in_last_chunk_ = FirstChunkCapacity();
      } else {
        // Stack is empty now.
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
      const auto& new_chunk = AppendHeapChunk();
      current += new_chunk.capacity;
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
  auto capacity() const -> int32_t { return GetSumCapacity(1 + heap_.size()); }

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
    if (lhs.empty()) {
      return true;
    }
    {
      auto chunk = 0;
      const T* lhs_array = lhs.inline_.array();
      const T* rhs_array = rhs.inline_.array();
      int32_t cap = FirstChunkCapacity();
      auto size = chunk < lhs.used_chunks_ - 1 ? cap : lhs.size_in_last_chunk_;
      if (!std::equal(lhs_array, lhs_array + size, rhs_array)) {
        return false;
      }
    }
    for (auto chunk = 1; chunk < lhs.used_chunks_; ++chunk) {
      const T* lhs_array = lhs.heap_[chunk - 1].array;
      const T* rhs_array = rhs.heap_[chunk - 1].array;
      int32_t cap = lhs.heap_[chunk - 1].capacity;
      CARBON_CHECK(cap == rhs.heap_[chunk - 1].capacity);
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
  // The capacity of the first chunk.
  static constexpr auto FirstChunkCapacity() -> int32_t {
    return 2;  // FIXME. Also document that there's inline storage on the class.
  }
  // The total capacity of the first `num_chunks` chunks.
  static constexpr auto GetSumCapacity(int32_t num_chunks) -> int32_t {
    return ((1 << num_chunks) - 1) * FirstChunkCapacity();
  }
  // Performs the inverse of GetSumCapacity.
  static constexpr auto NumChunksNeededForCapacity(int32_t cap) -> int32_t {
    // Given this formula, we solve for `num_chunks`.
    // cap = ((1 << num_chunks) - 1) * FirstChunkCapacity();
    auto ucap = static_cast<uint32_t>(cap);
    return std::bit_width(
        ((ucap + (FirstChunkCapacity() - 1)) / FirstChunkCapacity()));
  }

  struct InlineChunk {
    alignas(alignof(T)) char byte_array[FirstChunkCapacity() * sizeof(T)];
    auto array() -> T* { return reinterpret_cast<T*>(byte_array); }
    auto array() const -> const T* {
      return reinterpret_cast<const T*>(byte_array);
    }

    InlineChunk() {
      // TODO: Remove the #if when we move to C++23.
#if __has_cpp_attribute(__cpp_lib_start_lifetime_as)
      std::start_lifetime_as_array<T>(byte_array, FirstChunkCapacity());
#endif
      __asan_poison_memory_region(byte_array, FirstChunkCapacity() * sizeof(T));
    }
  };
  struct Chunk {
    T* array [[clang::require_explicit_initialization]];
    int32_t capacity [[clang::require_explicit_initialization]];
  };

  // Copy-constructs elements into this stack from `rhs`.
  auto CopyConstructFrom(const StableStack& rhs) -> void {
    if (rhs.empty()) {
      return;
    }

    {
      auto chunk = 0;
      auto rhs_array = rhs.inline_.array();
      auto cap = FirstChunkCapacity();
      T* array = inline_.array();
      auto size = chunk < used_chunks_ - 1 ? cap : size_in_last_chunk_;
      __asan_unpoison_memory_region(array, size * sizeof(T));
      std::uninitialized_copy_n(rhs_array, size, array);
    }

    // Reserve one less to account for the inline chunk.
    heap_.reserve(used_chunks_ - 1);
    for (auto chunk = 1; chunk < used_chunks_; ++chunk) {
      auto rhs_array = rhs.heap_[chunk - 1].array;
      auto cap = rhs.heap_[chunk - 1].capacity;
      T* array = std::allocator<T>().allocate(cap);
      heap_.push_back({.array = array, .capacity = cap});
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

    if (used_chunks == 0) {
      return;
    }

    {
      auto chunk = 0;
      auto size =
          chunk < used_chunks - 1 ? FirstChunkCapacity() : size_in_last_chunk;
      std::destroy_n(inline_.array(), size);
      __asan_poison_memory_region(inline_.array(),
                                  FirstChunkCapacity() * sizeof(T));
    }

    auto chunk = 1;
    for (; chunk < used_chunks; ++chunk) {
      auto* array = heap_[chunk - 1].array;
      auto cap = heap_[chunk - 1].capacity;
      auto size = chunk < used_chunks - 1 ? cap : size_in_last_chunk;
      std::destroy_n(array, size);
      std::allocator<T>().deallocate(array, cap);
    }
    for (; chunk < static_cast<int32_t>(heap_.size() - 1); ++chunk) {
      auto* array = heap_[chunk - 1].array;
      auto cap = heap_[chunk - 1].capacity;
      std::allocator<T>().deallocate(array, cap);
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
  auto Get(int32_t chunk, int32_t index) -> T& {
    CARBON_DCHECK(chunk >= 0);
    CARBON_DCHECK(chunk < used_chunks_);
    CARBON_DCHECK(index >= 0);
    auto cap = chunk == 0 ? FirstChunkCapacity() : heap_[chunk - 1].capacity;
    auto size = chunk == used_chunks_ - 1 ? size_in_last_chunk_ : cap;
    CARBON_DCHECK(index < size,
                  "Getting index {0} past end of last chunk with size {1}",
                  index, size);
    if (chunk == 0) {
      return inline_.array()[index];
    } else {
      return heap_[chunk - 1].array[index];
    }
  }
  auto Get(int32_t chunk, int32_t index) const -> const T& {
    return const_cast<StableStack*>(this)->Get(chunk, index);
  }

  // Allocate memory as needed to get a pointer that we can construct `T` into.
  // The pointer returned is still ASAN poisoned.
  auto PushSlot() -> T* {
    if (used_chunks_ == 0) {
      // Activate the `inline_` chunk and add a slot to it.
      T* slot = inline_.array();
      used_chunks_ = 1;
      size_in_last_chunk_ = 1;
      return slot;
    }

    if (used_chunks_ == 1 && size_in_last_chunk_ < FirstChunkCapacity()) {
      // Add a slot in the `inline_` chunk.
      T* slot = inline_.array() + size_in_last_chunk_;
      ++size_in_last_chunk_;
      return slot;
    }

    int32_t last_chunk = used_chunks_ - 1;
    if (last_chunk >= 1 &&
        size_in_last_chunk_ < heap_[last_chunk - 1].capacity) {
      // Add a slot in the last used `heap_` chunk.
      T* slot = heap_[last_chunk - 1].array + size_in_last_chunk_;
      ++size_in_last_chunk_;
      return slot;
    }

    if (used_chunks_ - 1 < static_cast<int32_t>(heap_.size())) {
      // Add a slot in an unused `heap_` chunk that already exists.
      T* slot = heap_[used_chunks_ - 1].array;
      ++used_chunks_;
      size_in_last_chunk_ = 1;
      return slot;
    }

    // Add a slot in a new `heap_` chunk.
    auto& new_chunk = AppendHeapChunk();
    ++used_chunks_;
    size_in_last_chunk_ = 1;
    return new_chunk.array;
  }
  auto AppendHeapChunk() -> Chunk& {
    auto last_capacity =
        heap_.empty() ? FirstChunkCapacity() : heap_.back().capacity;
    // We use a growth factor of 2 so that we can do bitshifts in
    // GetSumCapacity() to compute total capacity in O(1) time.
    auto next_cap = last_capacity * 2;
    T* array = std::allocator<T>().allocate(next_cap);
    __asan_poison_memory_region(array, next_cap * sizeof(T));
    heap_.push_back({.array = array, .capacity = next_cap});
    return heap_.back();
  }

  InlineChunk inline_;
  llvm::SmallVector<Chunk, 0> heap_;
  // The number of chunks that are being used, which have at least one value
  // stored in them. The first chunk is always `inline_`, so `used_chunks_ - 1`
  // gives the number of used chunks in `heap_`.
  int32_t used_chunks_ = 0;
  // The number of values in the last used chunk, either `inline_` or from
  // `heap_`.
  int32_t size_in_last_chunk_ = 0;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_STABLE_STACK_H_
