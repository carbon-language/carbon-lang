//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class ForwardIterator, class T>
//   void
//   uninitialized_fill(ForwardIterator first, ForwardIterator last,
//                      const T& x);

#include <memory>
#include <cassert>

#include "test_macros.h"

struct B
{
    static int count_;
    static int population_;
    int data_;
    explicit B() : data_(1) { ++population_; }
    B(const B &b) {
      ++count_;
      if (count_ == 3)
        TEST_THROW(1);
      data_ = b.data_;
      ++population_;
    }
    ~B() {data_ = 0; --population_; }
};

int B::count_ = 0;
int B::population_ = 0;

struct Nasty
{
    Nasty() : i_ ( counter_++ ) {}
    Nasty * operator &() const { return NULL; }
    int i_;
    static int counter_;
};

int Nasty::counter_ = 0;

int main(int, char**)
{
    {
    const int N = 5;
    char pool[sizeof(B)*N] = {0};
    B* bp = (B*)pool;
    assert(B::population_ == 0);
#ifndef TEST_HAS_NO_EXCEPTIONS
    try
    {
        std::uninitialized_fill(bp, bp+N, B());
        assert(false);
    }
    catch (...)
    {
        assert(B::population_ == 0);
    }
#endif
    B::count_ = 0;
    std::uninitialized_fill(bp, bp+2, B());
    for (int i = 0; i < 2; ++i)
        assert(bp[i].data_ == 1);
    assert(B::population_ == 2);
    }
    {
    const int N = 5;
    char pool[N*sizeof(Nasty)] = {0};
    Nasty* bp = (Nasty*)pool;

    Nasty::counter_ = 23;
    std::uninitialized_fill(bp, bp+N, Nasty());
    for (int i = 0; i < N; ++i)
        assert(bp[i].i_ == 23);
    }

  return 0;
}
