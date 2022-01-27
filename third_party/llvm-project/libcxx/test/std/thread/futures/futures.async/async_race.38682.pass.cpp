//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++03

// There's currently no release of OS X whose dylib contains the patch for
// PR38682. Since the fix for future<void> is in the dylib, this test may fail.
// UNSUPPORTED: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14}}

// This test is designed to cause and allow TSAN to detect a race condition
// in std::async, as reported in https://llvm.org/PR38682.

#include <cassert>
#include <functional>
#include <future>
#include <numeric>
#include <vector>

#include "test_macros.h"


static int worker(std::vector<int> const& data) {
  return std::accumulate(data.begin(), data.end(), 0);
}

static int& worker_ref(int& i) { return i; }

static void worker_void() { }

int main(int, char**) {
  // future<T>
  {
    std::vector<int> const v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    for (int i = 0; i != 20; ++i) {
      std::future<int> fut = std::async(std::launch::async, worker, v);
      int answer = fut.get();
      assert(answer == 55);
    }
  }

  // future<T&>
  {
    for (int i = 0; i != 20; ++i) {
      std::future<int&> fut = std::async(std::launch::async, worker_ref, std::ref(i));
      int& answer = fut.get();
      assert(answer == i);
    }
  }

  // future<void>
  {
    for (int i = 0; i != 20; ++i) {
      std::future<void> fut = std::async(std::launch::async, worker_void);
      fut.get();
    }
  }

  return 0;
}
