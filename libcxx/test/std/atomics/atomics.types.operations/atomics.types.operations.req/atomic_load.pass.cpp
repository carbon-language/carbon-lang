//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// XFAIL: !non-lockfree-atomics
//  ... assertion fails line 35

// <atomic>

// template <class T>
//     T
//     atomic_load(const volatile atomic<T>* obj);
//
// template <class T>
//     T
//     atomic_load(const atomic<T>* obj);

#include <atomic>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "atomic_helpers.h"

template <class T>
struct TestFn {
  void operator()() const {
    typedef std::atomic<T> A;
    A t(T(1));
    assert(std::atomic_load(&t) == T(1));
    volatile A vt(T(2));
    assert(std::atomic_load(&vt) == T(2));
  }
};

int main(int, char**)
{
    TestEachAtomicType<TestFn>()();

  return 0;
}
