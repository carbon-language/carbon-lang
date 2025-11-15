// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/latch.h"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <vector>

namespace Carbon {
namespace {

// Basic test for the Latch.
TEST(LatchTest, Basic) {
  Latch latch;
  Latch::Handle handle = latch.Init();
  // Dropping a copy doesn't satisfy the latch.
  Latch::Handle handle_copy = handle;
  EXPECT_FALSE(std::move(handle_copy).Drop());
  // Can create more copies even after we start dropping.
  Latch::Handle handle_copy2 = handle;
  EXPECT_FALSE(std::move(handle_copy2).Drop());
  // Now drop the last handle.
  EXPECT_TRUE(std::move(handle).Drop());
}

// Tests that the on-zero callback is called.
TEST(LatchTest, OnZeroCallback) {
  Latch latch;
  bool called = false;
  Latch::Handle handle = latch.Init([&] { called = true; });
  Latch::Handle handle2 = handle;
  Latch::Handle handle3 = handle;

  EXPECT_FALSE(called);
  EXPECT_FALSE(std::move(handle).Drop());
  EXPECT_FALSE(called);
  EXPECT_FALSE(std::move(handle2).Drop());
  EXPECT_FALSE(called);
  EXPECT_TRUE(std::move(handle3).Drop());
  EXPECT_TRUE(called);
}

// Tests moving a handle.
TEST(LatchTest, MoveHandle) {
  Latch latch;
  bool called = false;
  Latch::Handle handle = latch.Init([&] { called = true; });
  Latch::Handle handle2 = std::move(handle);

  // Check that dropping the new handle satisfies the latch.
  EXPECT_FALSE(called);
  EXPECT_TRUE(std::move(handle2).Drop());
  EXPECT_TRUE(called);
}

// Test that creating and destroying a handle without dropping works.
TEST(LatchTest, Destructor) {
  Latch latch;
  bool called = false;
  Latch::Handle handle = latch.Init([&] { called = true; });
  {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    Latch::Handle handle2 = handle;
    EXPECT_FALSE(called);
  }
  EXPECT_FALSE(called);
  EXPECT_TRUE(std::move(handle).Drop());
  EXPECT_TRUE(called);
}

// Tests calling `Init` more than once.
TEST(LatchTest, Reuse) {
  Latch latch;
  bool called = false;
  Latch::Handle handle = latch.Init([&] { called = true; });
  Latch::Handle handle2 = handle;

  EXPECT_FALSE(called);
  EXPECT_FALSE(std::move(handle).Drop());
  EXPECT_FALSE(called);
  EXPECT_TRUE(std::move(handle2).Drop());
  EXPECT_TRUE(called);

  // Now initialize the latch again with a new closure.
  bool called2 = false;
  Latch::Handle handle3 = latch.Init([&] { called2 = true; });
  Latch::Handle handle4 = handle3;

  EXPECT_FALSE(called2);
  EXPECT_FALSE(std::move(handle3).Drop());
  EXPECT_FALSE(called2);
  EXPECT_TRUE(std::move(handle4).Drop());
  EXPECT_TRUE(called2);
}

// Tests the latch with multiple threads.
TEST(LatchTest, MultiThreaded) {
  Latch latch;
  std::atomic<int> counter = 0;
  bool called = false;
  constexpr int NumThreads = 5;

  // The `on_zero` callback will be executed by the last thread to drop its
  // handle.
  auto handle = latch.Init([&] {
    // Check that all threads have done their work.
    EXPECT_EQ(counter.load(), NumThreads);
    called = true;
  });

  std::vector<std::thread> threads;
  threads.reserve(NumThreads);
  for (int i = 0; i < NumThreads; ++i) {
    threads.emplace_back([&, handle_copy = handle] {
      // Each thread has its own copy of the handle.
      // Simulate some work.
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      counter++;
      // The handle is dropped when the thread exits.
    });
  }

  // Drop the main thread's handle.
  std::move(handle).Drop();

  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_TRUE(called);
}

}  // namespace
}  // namespace Carbon
