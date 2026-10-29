// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_STABLE_STACK_H_
#define CARBON_TOOLCHAIN_BASE_STABLE_STACK_H_

#include <cstdint>
#include <new>
#include <type_traits>

#include "common/check.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "toolchain/base/mem_usage.h"

namespace Carbon {

// A stack that never reallocates when growing so as to not invalidate pointers
// to objects that remain in the stack.
//
// As a stack, all insertions must be pushed onto the back of the stack, and
// doing so is an O(1) operation. Peeking or popping the back of the stack is
// also an O(1) operation. It is also possible to peek further into the stack,
// but at a cost proportional to the distance from the back.
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
template <class T>
class StableStack {
 public:
  StableStack() = default;
  ~StableStack() { DestroyAll(); }

  // Copying the stack.
  StableStack(const StableStack& rhs)
    requires(std::is_copy_constructible_v<T>)
      : size_(rhs.size_),
        used_chunks_(rhs.used_chunks_),
        size_in_last_chunk_(rhs.size_in_last_chunk_) {
    ConstructFrom(rhs, [&rhs](int32_t chunk, int32_t index) -> const T& {
      return rhs.Get(chunk, index);
    });
  }
  auto operator=(const StableStack& rhs) -> StableStack&
    requires(std::is_copy_constructible_v<T> && std::is_copy_assignable_v<T>)
  {
    if (this == &rhs) {
      return *this;
    }

    DestroyAll();
    chunks_.truncate(0);
    size_ = rhs.size_;
    used_chunks_ = rhs.used_chunks_;
    size_in_last_chunk_ = rhs.size_in_last_chunk_;
    ConstructFrom(rhs, [&rhs](int32_t chunk, int32_t index) -> const T& {
      return rhs.Get(chunk, index);
    });
    return *this;
  }

  // Moving the stack. The moved-from `rhs` stack will be left empty with no
  // capacity.
  StableStack(StableStack&& rhs) noexcept
      : chunks_(std::exchange(rhs.chunks_, {})),
        size_(std::exchange(rhs.size_, 0)),
        used_chunks_(std::exchange(rhs.used_chunks_, 0)),
        size_in_last_chunk_(std::exchange(rhs.size_in_last_chunk_, 0)) {}
  auto operator=(StableStack&& rhs) noexcept -> StableStack& {
    if (this == &rhs) {
      return *this;
    }

    DestroyAll();
    chunks_ = std::exchange(rhs.chunks_, {});
    size_ = std::exchange(rhs.size_, 0);
    used_chunks_ = std::exchange(rhs.used_chunks_, 0);
    size_in_last_chunk_ = std::exchange(rhs.size_in_last_chunk_, 0);
    return *this;
  }

  // Returns a reference to an element in the stack. By default it returns the
  // most recently pushed element element at the back of the stack. But `n` may
  // be used to specify a distance from that element into the stack.
  //
  // This operation is O(1) when `n = 0` but otherwise has a cost proportional
  // to the size of `n`.
  auto Peek(int32_t n = 0) const -> const T& {
    CARBON_CHECK(!empty(), "Peek on an empty stack");
    auto [chunk, index] = GetIndexFromBack(n);
    return Get(chunk, index);
  }
  auto PeekMutable(int32_t n = 0) -> T& {
    CARBON_CHECK(!empty(), "Peek on an empty stack");
    auto [chunk, index] = GetIndexFromBack(n);
    return Get(chunk, index);
  }

  // Pushes a new element onto the back of the stack.
  auto Push(const T& t) {
    void* slot = PushSlot();
    __asan_unpoison_memory_region(slot, sizeof(T));
    std::construct_at<T>(reinterpret_cast<T*>(slot), t);
  }
  auto Push(T&& t) {
    void* slot = PushSlot();
    __asan_unpoison_memory_region(slot, sizeof(T));
    std::construct_at<T>(reinterpret_cast<T*>(slot), std::move(t));
  }

  // Removes the most recently pushed element from the back of the stack and
  // returns it.
  auto Pop() -> T {
    CARBON_CHECK(!empty(), "Pop on an empty stack");
    auto [chunk, index] = GetIndexAtBack();
    auto& ref = Get(chunk, index);

    size_--;
    size_in_last_chunk_--;
    if (size_in_last_chunk_ == 0) {
      used_chunks_--;
      if (used_chunks_ > 0) {
        size_in_last_chunk_ = chunks_[used_chunks_ - 1].capacity;
      }
    }

    auto ret = std::move(ref);
    std::destroy_at(&ref);
    __asan_poison_memory_region(&ref, sizeof(T));
    return ret;
  }

  // Ensure there is memory allocated to hold at least `reserve` elements.
  auto Reserve(int32_t reserve) {
    auto current = capacity();
    while (current < reserve) {
      auto capacity = NextChunkCapacity();
      void* array =
          ::operator new[](capacity * sizeof(T), std::align_val_t(alignof(T)));
      __asan_poison_memory_region(array, capacity * sizeof(T));
      chunks_.push_back({.array = array, .capacity = capacity});
      current += capacity;
    }
  }

  // Whether there are any elements pushed onto the stack.
  auto empty() const -> bool { return size_ == 0; }
  // The number of elements pushed onto the stack.
  auto size() const -> int32_t { return size_; }
  // Compute the capacity of the stack. The total capacity is not cached, so
  // this takes work proportional to the capacity of the stack.
  auto capacity() const {
    int32_t capacity = 0;
    for (const auto& chunk : chunks_) {
      capacity += chunk.capacity;
    }
    return capacity;
  }

  // Compares the contents of two stacks for equality.
  friend auto operator==(const StableStack& lhs, const StableStack& rhs) -> bool
    requires requires(const T& l, const T& r) {
      { l == r } -> std::convertible_to<bool>;
    }
  {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    if (lhs.empty()) {
      return true;
    }
    CARBON_CHECK(lhs.used_chunks_ == rhs.used_chunks_);
    for (auto chunk = 0; chunk < lhs.used_chunks_ - 1; ++chunk) {
      auto& l = lhs.chunks_[chunk];
      auto& r = rhs.chunks_[chunk];
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
      auto& l = lhs.chunks_[chunk];
      auto& r = rhs.chunks_[chunk];
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
    void* array [[clang::require_explicit_initialization]];
    int32_t capacity [[clang::require_explicit_initialization]];
  };

  // Constructs elements into this stack from `rhs`. Uses `getter` to get a
  // reference to each element in `rhs`.
  template <class Getter>
  auto ConstructFrom(const StableStack& rhs, Getter getter) {
    if (rhs.empty()) {
      return;
    }

    chunks_.reserve(used_chunks_);
    for (int chunk = 0; chunk < used_chunks_ - 1; ++chunk) {
      auto capacity = rhs.chunks_[chunk].capacity;
      void* array =
          ::operator new[](capacity * sizeof(T), std::align_val_t(alignof(T)));
      for (auto index = 0; index < capacity; ++index) {
        T* slot = reinterpret_cast<T*>(array) + index;
        std::construct_at(slot, getter(chunk, index));
      }
      chunks_.push_back({.array = array, .capacity = capacity});
    }
    {
      auto chunk = used_chunks_ - 1;
      auto capacity = rhs.chunks_[chunk].capacity;
      void* array =
          ::operator new[](capacity * sizeof(T), std::align_val_t(alignof(T)));
      __asan_poison_memory_region(
          reinterpret_cast<T*>(array) + size_in_last_chunk_,
          (capacity - size_in_last_chunk_) * sizeof(T));
      for (auto index = 0; index < size_in_last_chunk_; ++index) {
        T* slot = reinterpret_cast<T*>(array) + index;
        std::construct_at(slot, getter(chunk, index));
      }
      chunks_.push_back({.array = array, .capacity = capacity});
    }
  }

  // Destroy all the elements in the stack and free the arrays in the chunks.
  auto DestroyAll() {
    for (auto i = 0; i < used_chunks_ - 1; ++i) {
      std::destroy_n(reinterpret_cast<T*>(chunks_[i].array),
                     chunks_[i].capacity);
      ::operator delete[](chunks_[i].array, std::align_val_t(alignof(T)));
    }
    if (used_chunks_) {
      auto i = used_chunks_ - 1;
      std::destroy_n(reinterpret_cast<T*>(chunks_[i].array),
                     size_in_last_chunk_);
      ::operator delete[](chunks_[i].array, std::align_val_t(alignof(T)));
    }
    for (auto i = used_chunks_; i < static_cast<int32_t>(chunks_.size()); ++i) {
      ::operator delete[](chunks_[i].array, std::align_val_t(alignof(T)));
    }
  }

  // Get the index of the element at the back of the stack. This is an O(1)
  // operation.
  auto GetIndexAtBack() const
      -> std::pair<int32_t /*chunk*/, int32_t /*index*/> {
    int32_t chunk = used_chunks_ - 1;
    int32_t index = size_in_last_chunk_ - 1;
    return {chunk, index};
  }

  // Get the index of the element at distance `n` from the element at the back
  // of the stack. This is an O(n) operation.
  auto GetIndexFromBack(int32_t n) const
      -> std::pair<int32_t /*chunk*/, int32_t /*index*/> {
    auto [chunk, index] = GetIndexAtBack();
    // Walk through the chunks from the back, subtracting by the capacity, to
    // find the chunk that contains the position we are seeking.
    while (n) {
      auto step = std::min(n, index + 1);
      index -= step;
      if (index == -1) {
        CARBON_CHECK(chunk > 0, "Peek before the start of the stack");
        chunk--;
        index = chunks_[chunk].capacity - 1;
      }
      n -= step;
    }
    return {chunk, index};
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
    return *std::launder(reinterpret_cast<T*>(chunks_[chunk].array) + index);
  }

  // Allocate memory as needed to get a pointer that we can construct `T` into.
  // The pointer returned is still ASAN poisoned.
  auto PushSlot() -> void* {
    if (empty()) {
      auto& chunk = AddChunk();
      size_ = size_in_last_chunk_ = 1;
      return chunk.array;
    }
    int32_t chunk = used_chunks_ - 1;
    if (size_in_last_chunk_ == chunks_[chunk].capacity) {
      auto& chunk = AddChunk();
      size_in_last_chunk_ = 1;
      size_++;
      return chunk.array;
    } else {
      int32_t index = size_in_last_chunk_;
      size_in_last_chunk_++;
      size_++;
      return static_cast<T*>(chunks_[chunk].array) + index;
    }
  }

  // Append a new chunk to the vector of chunks. Each new chunk has a larger
  // capacity than the last. The chunk is initially fully ASAN poisoned.
  auto AddChunk() -> Chunk& {
    ++used_chunks_;
    if (used_chunks_ > static_cast<int32_t>(chunks_.size())) {
      auto capacity = NextChunkCapacity();
      void* array =
          ::operator new[](capacity * sizeof(T), std::align_val_t(alignof(T)));
      __asan_poison_memory_region(array, capacity * sizeof(T));
      chunks_.push_back({.array = array, .capacity = capacity});
    }
    return chunks_[used_chunks_ - 1];
  }

  auto NextChunkCapacity() {
    if (chunks_.empty()) {
      // The size of the first chunk.
      return 16;
    }
    // The growth factor from each chunk to the next. 2 matches the growth
    // factor of llvm::SmallVector.
    // https://github.com/llvm/llvm-project/blob/52be14d081f369c826a54c949b55eb1e1ba8c68c/llvm/lib/Support/SmallVector.cpp#L107
    return chunks_.back().capacity * 2;
  }

  llvm::SmallVector<Chunk, 1> chunks_;
  // The total number of values in the stack.
  int32_t size_ = 0;
  // The number of elements in `chunks_` that are being used, which have at
  // least one value stored in them.
  int32_t used_chunks_ = 0;
  // The number of values in the last used chunk in `chunks_`.
  int32_t size_in_last_chunk_ = 0;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_STABLE_STACK_H_
