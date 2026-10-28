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

template <class T>
class StableStack;

template <class T>
class StableStack {
 public:
  StableStack() = default;
  ~StableStack() { DestroyAll(); }

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
    requires(std::is_copy_constructible_v<T>)
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

  StableStack(StableStack&& rhs) noexcept
    requires(std::is_move_constructible_v<T>)
      : size_(rhs.size_),
        used_chunks_(rhs.used_chunks_),
        size_in_last_chunk_(rhs.size_in_last_chunk_) {
    ConstructFrom(rhs, [&rhs](int32_t chunk, int32_t index) -> T&& {
      return std::move(rhs.Get(chunk, index));
    });
  }
  auto operator=(StableStack&& rhs) -> StableStack&
    requires(std::is_move_constructible_v<T>)
  {
    if (this == &rhs) {
      return *this;
    }

    DestroyAll();
    chunks_.truncate(0);
    size_ = rhs.size_;
    used_chunks_ = rhs.used_chunks_;
    size_in_last_chunk_ = rhs.size_in_last_chunk_;
    ConstructFrom(rhs, [&rhs](int32_t chunk, int32_t index) -> T&& {
      return std::move(rhs.Get(chunk, index));
    });
    return *this;
  }

  auto Peek(int32_t n = 0) const -> const T& {
    CARBON_CHECK(!empty(), "Peek on an empty stack");
    auto [chunk, index] = GetIndexFromBack(n);
    return Get(chunk, index);
  }

  auto PeekMutable(int32_t n = 0) const -> T& {
    CARBON_CHECK(!empty(), "Peek on an empty stack");
    auto [chunk, index] = GetIndexFromBack(n);
    return Get(chunk, index);
  }

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
    return std::move(ret);
  }

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

  auto empty() const -> bool { return size_ == 0; }
  auto size() const -> int32_t { return size_; }
  auto capacity() const {
    int32_t capacity = 0;
    for (const auto& chunk : chunks_) {
      capacity += chunk.capacity;
    }
    return capacity;
  }

  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Add(label.str(), size(), capacity());
  }

 private:
  struct Chunk {
    void* array [[clang::require_explicit_initialization]];
    int32_t capacity [[clang::require_explicit_initialization]];
  };

  template <class Getter>
  auto ConstructFrom(const StableStack& rhs, Getter getter) {
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

  auto DestroyAll() {
    if (empty()) {
      return;
    }

    for (auto i = 0; i < used_chunks_ - 1; ++i) {
      std::destroy_n(reinterpret_cast<T*>(chunks_[i].array),
                     chunks_[i].capacity);
      ::operator delete[](chunks_[i].array, std::align_val_t(alignof(T)));
    }
    {
      auto i = used_chunks_ - 1;
      std::destroy_n(reinterpret_cast<T*>(chunks_[i].array),
                     size_in_last_chunk_);
      ::operator delete[](chunks_[i].array, std::align_val_t(alignof(T)));
    }
    for (auto i = used_chunks_; i < static_cast<int32_t>(chunks_.size()); ++i) {
      ::operator delete[](chunks_[i].array, std::align_val_t(alignof(T)));
    }
  }

  auto GetIndexAtBack() const
      -> std::pair<int32_t /*chunk*/, int32_t /*index*/> {
    int32_t chunk = used_chunks_ - 1;
    int32_t index = size_in_last_chunk_ - 1;
    return {chunk, index};
  }

  auto GetIndexFromBack(int32_t n) const
      -> std::pair<int32_t /*chunk*/, int32_t /*index*/> {
    auto [chunk, index] = GetIndexAtBack();
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
    if (empty()) {
      return 16;
    }
    return chunks_.back().capacity * 3;
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
