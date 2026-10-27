// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/stable_stack.h"

#include <gtest/gtest.h>

namespace Carbon {
namespace {

TEST(StableStack, DestroyEmpty) {
  llvm::BumpPtrAllocator alloc;
  StableStack<int> stack(alloc);
  EXPECT_EQ(stack.empty(), true);
}

TEST(StableStack, Growth) {
  llvm::BumpPtrAllocator alloc;
  StableStack<int> stack(alloc);
  EXPECT_EQ(stack.empty(), true);
  EXPECT_EQ(stack.size(), 0);
  EXPECT_EQ(stack.capacity(), 0);

  int i = 0;
  stack.Push(i++);
  EXPECT_EQ(stack.empty(), false);
  EXPECT_EQ(stack.size(), i);
  EXPECT_GE(stack.capacity(), 1);

  auto capacity = stack.capacity();
  while (stack.capacity() == capacity) {
    stack.Push(i++);
    EXPECT_EQ(stack.empty(), false);
    EXPECT_EQ(stack.size(), i);
  }

  capacity = stack.capacity();
  while (stack.capacity() == capacity) {
    stack.Push(i++);
    EXPECT_EQ(stack.empty(), false);
    EXPECT_EQ(stack.size(), i);
  }
}

template <typename T, typename F>
static auto GrowCapacityTimes(StableStack<T>& stack, int n, F factory) {
  while (n) {
    auto capacity = stack.capacity();
    while (stack.capacity() == capacity) {
      stack.Push(factory());
    }
    --n;
  }
}

TEST(StableStack, PushPop) {
  llvm::BumpPtrAllocator alloc;
  StableStack<int> stack(alloc);
  int i = 0;
  GrowCapacityTimes(stack, 3, [&i]() { return i++; });
  EXPECT_EQ(stack.size(), i);

  stack.Push(i++);
  EXPECT_EQ(stack.size(), i);

  EXPECT_EQ(stack.Pop(), --i);
  EXPECT_EQ(stack.size(), i);

  // Drops below the last capacity jump.
  EXPECT_EQ(stack.Pop(), --i);
  EXPECT_EQ(stack.size(), i);

  stack.Push(i++);
  EXPECT_EQ(stack.size(), i);

  while (i) {
    EXPECT_EQ(stack.Pop(), --i);
    EXPECT_EQ(stack.size(), i);
  }
  EXPECT_EQ(stack.empty(), true);
}

TEST(StableStack, EmptyPushPop) {
  llvm::BumpPtrAllocator alloc;
  StableStack<int> stack(alloc);

  stack.Push(1);
  EXPECT_EQ(stack.size(), 1);
  EXPECT_EQ(stack.empty(), false);

  EXPECT_EQ(stack.Pop(), 1);
  EXPECT_EQ(stack.size(), 0);
  EXPECT_EQ(stack.empty(), true);

  stack.Push(2);
  EXPECT_EQ(stack.size(), 1);
  EXPECT_EQ(stack.empty(), false);

  EXPECT_EQ(stack.Pop(), 2);
  EXPECT_EQ(stack.size(), 0);
  EXPECT_EQ(stack.empty(), true);
}

TEST(StableStack, CopyConstructDestroy) {
  int constructed = 0;
  int destroyed = 0;
  class Counter {
   public:
    Counter(int* constructed, int* destroyed)
        : constructed_(constructed), destroyed_(destroyed) {}
    Counter(const Counter& rhs)
        : constructed_(rhs.constructed_), destroyed_(rhs.destroyed_) {
      (*constructed_)++;
    }
    ~Counter() { (*destroyed_)++; }

   private:
    int* constructed_;
    int* destroyed_;
  };

  int size;
  {
    llvm::BumpPtrAllocator alloc;
    StableStack<Counter> stack(alloc);
    GrowCapacityTimes(stack, 3,
                      [&]() { return Counter(&constructed, &destroyed); });
    EXPECT_EQ(constructed, stack.size());
    // Each Counter constructed in the lambda was destroyed.
    EXPECT_EQ(destroyed, stack.size());
    size = stack.size();
    constructed = destroyed = 0;

    stack.Peek();
    EXPECT_EQ(constructed, 0);
    EXPECT_EQ(destroyed, 0);
    stack.PeekMutable();
    EXPECT_EQ(constructed, 0);
    EXPECT_EQ(destroyed, 0);
  }
  EXPECT_EQ(constructed, 0);
  // Each Counter held in the stack was destroyed.
  EXPECT_EQ(destroyed, size);
}

TEST(StableStack, MoveConstructDestroy) {
  int constructed = 0;
  int destroyed = 0;
  class Counter {
   public:
    Counter(int* constructed, int* destroyed)
        : constructed_(constructed), destroyed_(destroyed) {}
    Counter(Counter&& rhs)
        : constructed_(rhs.constructed_), destroyed_(rhs.destroyed_) {
      (*constructed_)++;
    }
    ~Counter() { (*destroyed_)++; }

   private:
    int* constructed_;
    int* destroyed_;
  };

  int size;
  {
    llvm::BumpPtrAllocator alloc;
    StableStack<Counter> stack(alloc);
    GrowCapacityTimes(stack, 3,
                      [&]() { return Counter(&constructed, &destroyed); });
    EXPECT_EQ(constructed, stack.size());
    // Each Counter constructed in the lambda was destroyed.
    EXPECT_EQ(destroyed, stack.size());
    size = stack.size();
    constructed = destroyed = 0;

    stack.Peek();
    EXPECT_EQ(constructed, 0);
    EXPECT_EQ(destroyed, 0);
    stack.PeekMutable();
    EXPECT_EQ(constructed, 0);
    EXPECT_EQ(destroyed, 0);
  }
  EXPECT_EQ(constructed, 0);
  // Each Counter held in the stack was destroyed.
  EXPECT_EQ(destroyed, size);
}

TEST(StableStack, CopyStack) {
  int constructed = 0;
  int destroyed = 0;
  class Counter {
   public:
    Counter(int value, int* constructed, int* destroyed)
        : value_(value), constructed_(constructed), destroyed_(destroyed) {}
    Counter(const Counter& rhs)
        : value_(rhs.value_),
          constructed_(rhs.constructed_),
          destroyed_(rhs.destroyed_) {
      (*constructed_)++;
    }
    ~Counter() { (*destroyed_)++; }

    auto operator==(const Counter& rhs) const -> bool {
      return value_ == rhs.value_;
    }

   private:
    int value_;
    int* constructed_;
    int* destroyed_;
  };

  int size;
  {
    llvm::BumpPtrAllocator alloc;
    StableStack<Counter> stack(alloc);
    GrowCapacityTimes(stack, 3, [&, i = 0]() mutable {
      return Counter(i++, &constructed, &destroyed);
    });
    constructed = destroyed = 0;
    size = stack.size();

    // Copy construction.
    StableStack<Counter> stack2 = stack;
    EXPECT_EQ(constructed, size);
    EXPECT_EQ(destroyed, 0);
    constructed = destroyed = 0;

    EXPECT_EQ(stack.size(), stack2.size());
    EXPECT_LE(stack.capacity(), stack2.capacity());
    for (auto i = 0; i < size; ++i) {
      EXPECT_EQ(stack.Peek(i), stack2.Peek(i));
    }
    EXPECT_EQ(constructed, 0);
    EXPECT_EQ(destroyed, 0);
    constructed = destroyed = 0;

    // Self-assignment.
    stack2 = [&]() -> StableStack<Counter>& { return stack2; }();
    EXPECT_EQ(constructed, 0);
    EXPECT_EQ(destroyed, 0);
    constructed = destroyed = 0;

    StableStack<Counter> stack3(alloc);
    EXPECT_EQ(constructed, 0);
    EXPECT_EQ(destroyed, 0);
    constructed = destroyed = 0;

    // Copy assignment to empty.
    stack3 = stack2;
    EXPECT_EQ(constructed, size);
    EXPECT_EQ(destroyed, 0);
    constructed = destroyed = 0;

    // Copy assignment to non-empty. We destroy the LHS and reconstruct from the
    // RHS.
    stack3 = stack2;
    EXPECT_EQ(constructed, size);
    EXPECT_EQ(destroyed, size);
    constructed = destroyed = 0;
  }
  EXPECT_EQ(constructed, 0);
  // Each Counter held in each stack was destroyed.
  EXPECT_EQ(destroyed, size * 3);
}

TEST(StableStack, MoveStack) {
  int constructed = 0;
  int destroyed = 0;
  class Counter {
   public:
    Counter(int value, int* constructed, int* destroyed)
        : value_(value), constructed_(constructed), destroyed_(destroyed) {}
    Counter(Counter&& rhs)
        : value_(rhs.value_),
          constructed_(rhs.constructed_),
          destroyed_(rhs.destroyed_) {
      (*constructed_)++;
    }
    ~Counter() { (*destroyed_)++; }

    auto operator==(const Counter& rhs) const -> bool {
      return value_ == rhs.value_;
    }

   private:
    int value_;
    int* constructed_;
    int* destroyed_;
  };

  int size;
  {
    llvm::BumpPtrAllocator alloc;
    StableStack<Counter> stack(alloc);
    GrowCapacityTimes(stack, 3, [&, i = 0]() mutable {
      return Counter(i++, &constructed, &destroyed);
    });
    constructed = destroyed = 0;
    size = stack.size();

    StableStack<Counter> stack2 = std::move(stack);
    EXPECT_EQ(constructed, size);
    EXPECT_EQ(destroyed, 0);
    constructed = destroyed = 0;

    EXPECT_EQ(stack.size(), stack2.size());
    EXPECT_LE(stack.capacity(), stack2.capacity());
    for (auto i = 0; i < size; ++i) {
      EXPECT_EQ(stack.Peek(i), stack2.Peek(i));
    }
    EXPECT_EQ(constructed, 0);
    EXPECT_EQ(destroyed, 0);
    constructed = destroyed = 0;

    // Self-assignment.
    stack2 = [&]() -> StableStack<Counter>&& { return std::move(stack2); }();
    EXPECT_EQ(constructed, 0);
    EXPECT_EQ(destroyed, 0);
    constructed = destroyed = 0;

    StableStack<Counter> stack3(alloc);
    EXPECT_EQ(constructed, 0);
    EXPECT_EQ(destroyed, 0);
    constructed = destroyed = 0;

    // Move assignment to empty.
    stack3 = std::move(stack2);
    EXPECT_EQ(constructed, size);
    EXPECT_EQ(destroyed, 0);
    constructed = destroyed = 0;

    // Move assignment to non-empty. We destroy the LHS and reconstruct from the
    // RHS.
    stack3 = std::move(stack2);
    EXPECT_EQ(constructed, size);
    EXPECT_EQ(destroyed, size);
    constructed = destroyed = 0;
  }
  EXPECT_EQ(constructed, 0);
  // Each Counter held in each stack was destroyed.
  EXPECT_EQ(destroyed, size * 3);
}

TEST(StableStack, Peek) {
  llvm::BumpPtrAllocator alloc;
  StableStack<int> stack(alloc);
  stack.Push(1);

  decltype(auto) r = stack.Peek();
  static_assert(std::same_as<decltype(r), const int&>);

  decltype(auto) m = stack.PeekMutable();
  static_assert(std::same_as<decltype(m), int&>);

  stack.Push(2);
  EXPECT_EQ(stack.Peek(), 2);
  EXPECT_EQ(stack.PeekMutable(), 2);
  stack.PeekMutable() = 4;
  EXPECT_EQ(stack.Peek(), 4);
  EXPECT_EQ(stack.PeekMutable(), 4);
  EXPECT_EQ(stack.Pop(), 4);

  EXPECT_EQ(stack.Peek(), 1);
  EXPECT_EQ(stack.PeekMutable(), 1);
  stack.PeekMutable() = 3;
  EXPECT_EQ(stack.Peek(), 3);
  EXPECT_EQ(stack.PeekMutable(), 3);
  EXPECT_EQ(stack.Pop(), 3);
}

TEST(StableStack, Reserve) {
  llvm::BumpPtrAllocator alloc;
  StableStack<int> stack(alloc);
  EXPECT_EQ(stack.capacity(), 0);

  stack.Reserve(2);
  EXPECT_GE(stack.capacity(), 2);
  auto capacity = stack.capacity();
  stack.Reserve(1);
  EXPECT_EQ(stack.capacity(), capacity);
  stack.Reserve(2);
  EXPECT_EQ(stack.capacity(), capacity);

  stack.Reserve(1000);
  EXPECT_GE(stack.capacity(), 1000);
  EXPECT_GE(stack.size(), 0);
  EXPECT_GE(stack.empty(), true);

  capacity = stack.capacity();
  stack.Reserve(1);
  EXPECT_EQ(stack.capacity(), capacity);
  stack.Reserve(0);
  EXPECT_EQ(stack.capacity(), capacity);

  for (int i = 0; i < capacity; ++i) {
    stack.Push(i);
  }
  EXPECT_EQ(stack.capacity(), capacity);
  EXPECT_EQ(stack.size(), capacity);
}

}  // namespace
}  // namespace Carbon
