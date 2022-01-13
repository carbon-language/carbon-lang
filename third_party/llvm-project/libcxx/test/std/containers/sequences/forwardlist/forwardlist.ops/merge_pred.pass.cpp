//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// template <class Compare> void merge(forward_list&& x, Compare comp);

#include <forward_list>
#include <iterator>
#include <functional>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef int T;
        typedef std::forward_list<T> C;
        const T t1[] = {13, 12, 7, 6, 5, 3};
        const T t2[] = {15, 14, 11, 10, 9, 8, 4, 2, 1, 0};
        const T t3[] = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
        C c1(std::begin(t1), std::end(t1));
        C c2(std::begin(t2), std::end(t2));
        c1.merge(c2, std::greater<T>());
        C c3(std::begin(t3), std::end(t3));
        assert(c1 == c3);
    }
#if TEST_STD_VER >= 11
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        const T t1[] = {13, 12, 7, 6, 5, 3};
        const T t2[] = {15, 14, 11, 10, 9, 8, 4, 2, 1, 0};
        const T t3[] = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
        C c1(std::begin(t1), std::end(t1));
        C c2(std::begin(t2), std::end(t2));
        c1.merge(c2, std::greater<T>());
        C c3(std::begin(t3), std::end(t3));
        assert(c1 == c3);
    }
#endif

  return 0;
}
