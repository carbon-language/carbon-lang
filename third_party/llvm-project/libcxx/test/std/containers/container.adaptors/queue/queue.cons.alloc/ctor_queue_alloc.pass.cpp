//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// template <class Alloc>
//   queue(const queue& q, const Alloc& a);

#include <queue>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"

template <class C>
C
make(int n)
{
    C c;
    for (int i = 0; i < n; ++i)
        c.push_back(i);
    return c;
}

typedef std::deque<int, test_allocator<int> > C;

template <class T>
struct test
    : public std::queue<T, C>
{
    typedef std::queue<T, C> base;
    typedef test_allocator<int>      allocator_type;
    typedef typename base::container_type container_type;

    explicit test(const allocator_type& a) : base(a) {}
    test(const container_type& container, const allocator_type& a) : base(container, a) {}
    test(const test& q, const allocator_type& a) : base(q, a) {}
    allocator_type get_allocator() {return this->c.get_allocator();}
};

int main(int, char**)
{
    test<int> q(make<C>(5), test_allocator<int>(4));
    test<int> q2(q, test_allocator<int>(5));
    assert(q2.get_allocator() == test_allocator<int>(5));
    assert(q2.size() == 5);

  return 0;
}
