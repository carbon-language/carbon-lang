//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// void resize(size_type n);

#include <forward_list>
#include <cassert>

#include "test_macros.h"
#include "DefaultOnly.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef DefaultOnly T;
        typedef std::forward_list<T> C;
        C c;
        c.resize(0);
        assert(distance(c.begin(), c.end()) == 0);
        c.resize(10);
        assert(distance(c.begin(), c.end()) == 10);
        c.resize(20);
        assert(distance(c.begin(), c.end()) == 20);
        c.resize(5);
        assert(distance(c.begin(), c.end()) == 5);
        c.resize(0);
        assert(distance(c.begin(), c.end()) == 0);
    }
    {
        typedef int T;
        typedef std::forward_list<T> C;
        const T t[] = {0, 1, 2, 3, 4};
        C c(std::begin(t), std::end(t));

        c.resize(3);
        assert(distance(c.begin(), c.end()) == 3);
        assert(*next(c.begin(), 0) == 0);
        assert(*next(c.begin(), 1) == 1);
        assert(*next(c.begin(), 2) == 2);

        c.resize(6);
        assert(distance(c.begin(), c.end()) == 6);
        assert(*next(c.begin(), 0) == 0);
        assert(*next(c.begin(), 1) == 1);
        assert(*next(c.begin(), 2) == 2);
        assert(*next(c.begin(), 3) == 0);
        assert(*next(c.begin(), 4) == 0);
        assert(*next(c.begin(), 5) == 0);

        c.resize(6);
        assert(distance(c.begin(), c.end()) == 6);
        assert(*next(c.begin(), 0) == 0);
        assert(*next(c.begin(), 1) == 1);
        assert(*next(c.begin(), 2) == 2);
        assert(*next(c.begin(), 3) == 0);
        assert(*next(c.begin(), 4) == 0);
        assert(*next(c.begin(), 5) == 0);
    }
#if TEST_STD_VER >= 11
    {
        typedef DefaultOnly T;
        typedef std::forward_list<T, min_allocator<T>> C;
        C c;
        c.resize(0);
        assert(distance(c.begin(), c.end()) == 0);
        c.resize(10);
        assert(distance(c.begin(), c.end()) == 10);
        c.resize(20);
        assert(distance(c.begin(), c.end()) == 20);
        c.resize(5);
        assert(distance(c.begin(), c.end()) == 5);
        c.resize(0);
        assert(distance(c.begin(), c.end()) == 0);
    }
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        const T t[] = {0, 1, 2, 3, 4};
        C c(std::begin(t), std::end(t));

        c.resize(3);
        assert(distance(c.begin(), c.end()) == 3);
        assert(*next(c.begin(), 0) == 0);
        assert(*next(c.begin(), 1) == 1);
        assert(*next(c.begin(), 2) == 2);

        c.resize(6);
        assert(distance(c.begin(), c.end()) == 6);
        assert(*next(c.begin(), 0) == 0);
        assert(*next(c.begin(), 1) == 1);
        assert(*next(c.begin(), 2) == 2);
        assert(*next(c.begin(), 3) == 0);
        assert(*next(c.begin(), 4) == 0);
        assert(*next(c.begin(), 5) == 0);

        c.resize(6);
        assert(distance(c.begin(), c.end()) == 6);
        assert(*next(c.begin(), 0) == 0);
        assert(*next(c.begin(), 1) == 1);
        assert(*next(c.begin(), 2) == 2);
        assert(*next(c.begin(), 3) == 0);
        assert(*next(c.begin(), 4) == 0);
        assert(*next(c.begin(), 5) == 0);
    }
#endif

  return 0;
}
