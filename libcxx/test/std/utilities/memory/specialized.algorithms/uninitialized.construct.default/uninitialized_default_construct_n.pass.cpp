//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <memory>

// template <class ForwardIt>
// void uninitialized_default_construct(ForwardIt, ForwardIt);

#include <memory>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

struct Counted {
  static int count;
  static int constructed;
  static void reset() { count = constructed =  0; }
  explicit Counted() { ++count; ++constructed; }
  Counted(Counted const&) { assert(false); }
  ~Counted() { assert(count > 0); --count; }
  friend void operator&(Counted) = delete;
};
int Counted::count = 0;
int Counted::constructed = 0;


struct ThrowsCounted {
  static int count;
  static int constructed;
  static int throw_after;
  static void reset() { throw_after = count = constructed =  0; }
  explicit ThrowsCounted() {
      ++constructed;
      if (throw_after > 0 && --throw_after == 0) {
          TEST_THROW(1);
      }
      ++count;
  }
  ThrowsCounted(ThrowsCounted const&) { assert(false); }
  ~ThrowsCounted() { assert(count > 0); --count; }
  friend void operator&(ThrowsCounted) = delete;
};
int ThrowsCounted::count = 0;
int ThrowsCounted::constructed = 0;
int ThrowsCounted::throw_after = 0;

void test_ctor_throws()
{
#ifndef TEST_HAS_NO_EXCEPTIONS
    using It = forward_iterator<ThrowsCounted*>;
    const int N = 5;
    alignas(ThrowsCounted) char pool[sizeof(ThrowsCounted)*N] = {};
    ThrowsCounted* p = (ThrowsCounted*)pool;
    try {
        ThrowsCounted::throw_after = 4;
        std::uninitialized_default_construct_n(It(p), N);
        assert(false);
    } catch (...) {}
    assert(ThrowsCounted::count == 0);
    assert(ThrowsCounted::constructed == 4); // forth construction throws
#endif
}

void test_counted()
{
    using It = forward_iterator<Counted*>;
    const int N = 5;
    alignas(Counted) char pool[sizeof(Counted)*N] = {};
    Counted* p = (Counted*)pool;
    It e = std::uninitialized_default_construct_n(It(p), 1);
    assert(e == It(p+1));
    assert(Counted::count == 1);
    assert(Counted::constructed == 1);
    e = std::uninitialized_default_construct_n(It(p+1), 4);
    assert(e == It(p+N));
    assert(Counted::count == 5);
    assert(Counted::constructed == 5);
    std::destroy(p, p+N);
    assert(Counted::count == 0);
}

void test_value_initialized()
{
    using It = forward_iterator<int*>;
    const int N = 5;
    int pool[N] = {-1, -1, -1, -1, -1};
    int* p = pool;
    auto e = std::uninitialized_default_construct_n(It(p), 1);
    assert(e == It(p+1));
    assert(pool[0] == -1);
    assert(pool[1] == -1);
    e = std::uninitialized_default_construct_n(It(p+1), 4);
    assert(e == It(p+N));
    assert(pool[1] == -1);
    assert(pool[2] == -1);
    assert(pool[3] == -1);
    assert(pool[4] == -1);
}


int main(int, char**)
{
    test_counted();
    test_value_initialized();
    test_ctor_throws();

  return 0;
}
