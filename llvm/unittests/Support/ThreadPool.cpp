//========- unittests/Support/ThreadPools.cpp - ThreadPools.h tests --========//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ThreadPool.h"

#include "llvm/ADT/STLExtras.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace std::chrono;

/// Try best to make this thread not progress faster than the main thread
static void yield() {
#ifdef LLVM_ENABLE_THREADS
  std::this_thread::yield();
#endif
  std::this_thread::sleep_for(milliseconds(200));
#ifdef LLVM_ENABLE_THREADS
  std::this_thread::yield();
#endif
}

TEST(ThreadPoolTest, AsyncBarrier) {
  // test that async & barrier work together properly.

  std::atomic_int checked_in{0};

  ThreadPool Pool;
  for (size_t i = 0; i < 5; ++i) {
    Pool.async([&checked_in, i] {
      yield();
      ++checked_in;
    });
  }
  ASSERT_EQ(0, checked_in);
  Pool.wait();
  ASSERT_EQ(5, checked_in);
}

TEST(ThreadPoolTest, Async) {
  ThreadPool Pool;
  std::atomic_int i{0};
  // sleep here just to ensure that the not-equal is correct.
  Pool.async([&i] {
    yield();
    ++i;
  });
  Pool.async([&i] { ++i; });
  ASSERT_NE(2, i.load());
  Pool.wait();
  ASSERT_EQ(2, i.load());
}

TEST(ThreadPoolTest, GetFuture) {
  ThreadPool Pool;
  std::atomic_int i{0};
  // sleep here just to ensure that the not-equal is correct.
  Pool.async([&i] {
    yield();
    ++i;
  });
  // Force the future using get()
  Pool.async([&i] { ++i; }).get();
  ASSERT_NE(2, i.load());
  Pool.wait();
  ASSERT_EQ(2, i.load());
}

TEST(ThreadPoolTest, PoolDestruction) {
  // Test that we are waiting on destruction
  std::atomic_int checked_in{0};

  {
    ThreadPool Pool;
    for (size_t i = 0; i < 5; ++i) {
      Pool.async([&checked_in, i] {
        yield();
        ++checked_in;
      });
    }
    ASSERT_EQ(0, checked_in);
  }
  ASSERT_EQ(5, checked_in);
}
