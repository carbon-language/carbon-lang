//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// iterator erase_after(const_iterator first, const_iterator last);

#include <forward_list>
#include <cassert>
#include <iterator>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef int T;
        typedef std::forward_list<T> C;
        const T t[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        C c(std::begin(t), std::end(t));

        C::iterator i = c.erase_after(std::next(c.cbefore_begin(), 4), std::next(c.cbefore_begin(), 4));
        assert(i == std::next(c.cbefore_begin(), 4));
        assert(std::distance(c.begin(), c.end()) == 10);
        assert(*std::next(c.begin(), 0) == 0);
        assert(*std::next(c.begin(), 1) == 1);
        assert(*std::next(c.begin(), 2) == 2);
        assert(*std::next(c.begin(), 3) == 3);
        assert(*std::next(c.begin(), 4) == 4);
        assert(*std::next(c.begin(), 5) == 5);
        assert(*std::next(c.begin(), 6) == 6);
        assert(*std::next(c.begin(), 7) == 7);
        assert(*std::next(c.begin(), 8) == 8);
        assert(*std::next(c.begin(), 9) == 9);

        i = c.erase_after(std::next(c.cbefore_begin(), 2), std::next(c.cbefore_begin(), 5));
        assert(i == std::next(c.begin(), 2));
        assert(std::distance(c.begin(), c.end()) == 8);
        assert(*std::next(c.begin(), 0) == 0);
        assert(*std::next(c.begin(), 1) == 1);
        assert(*std::next(c.begin(), 2) == 4);
        assert(*std::next(c.begin(), 3) == 5);
        assert(*std::next(c.begin(), 4) == 6);
        assert(*std::next(c.begin(), 5) == 7);
        assert(*std::next(c.begin(), 6) == 8);
        assert(*std::next(c.begin(), 7) == 9);

        i = c.erase_after(std::next(c.cbefore_begin(), 2), std::next(c.cbefore_begin(), 3));
        assert(i == std::next(c.begin(), 2));
        assert(std::distance(c.begin(), c.end()) == 8);
        assert(*std::next(c.begin(), 0) == 0);
        assert(*std::next(c.begin(), 1) == 1);
        assert(*std::next(c.begin(), 2) == 4);
        assert(*std::next(c.begin(), 3) == 5);
        assert(*std::next(c.begin(), 4) == 6);
        assert(*std::next(c.begin(), 5) == 7);
        assert(*std::next(c.begin(), 6) == 8);
        assert(*std::next(c.begin(), 7) == 9);

        i = c.erase_after(std::next(c.cbefore_begin(), 5), std::next(c.cbefore_begin(), 9));
        assert(i == c.end());
        assert(std::distance(c.begin(), c.end()) == 5);
        assert(*std::next(c.begin(), 0) == 0);
        assert(*std::next(c.begin(), 1) == 1);
        assert(*std::next(c.begin(), 2) == 4);
        assert(*std::next(c.begin(), 3) == 5);
        assert(*std::next(c.begin(), 4) == 6);

        i = c.erase_after(std::next(c.cbefore_begin(), 0), std::next(c.cbefore_begin(), 2));
        assert(i == c.begin());
        assert(std::distance(c.begin(), c.end()) == 4);
        assert(*std::next(c.begin(), 0) == 1);
        assert(*std::next(c.begin(), 1) == 4);
        assert(*std::next(c.begin(), 2) == 5);
        assert(*std::next(c.begin(), 3) == 6);

        i = c.erase_after(std::next(c.cbefore_begin(), 0), std::next(c.cbefore_begin(), 5));
        assert(i == c.begin());
        assert(i == c.end());
        assert(std::distance(c.begin(), c.end()) == 0);
    }
#if TEST_STD_VER >= 11
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        const T t[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        C c(std::begin(t), std::end(t));

        C::iterator i = c.erase_after(std::next(c.cbefore_begin(), 4), std::next(c.cbefore_begin(), 4));
        assert(i == std::next(c.cbefore_begin(), 4));
        assert(std::distance(c.begin(), c.end()) == 10);
        assert(*std::next(c.begin(), 0) == 0);
        assert(*std::next(c.begin(), 1) == 1);
        assert(*std::next(c.begin(), 2) == 2);
        assert(*std::next(c.begin(), 3) == 3);
        assert(*std::next(c.begin(), 4) == 4);
        assert(*std::next(c.begin(), 5) == 5);
        assert(*std::next(c.begin(), 6) == 6);
        assert(*std::next(c.begin(), 7) == 7);
        assert(*std::next(c.begin(), 8) == 8);
        assert(*std::next(c.begin(), 9) == 9);

        i = c.erase_after(std::next(c.cbefore_begin(), 2), std::next(c.cbefore_begin(), 5));
        assert(i == std::next(c.begin(), 2));
        assert(std::distance(c.begin(), c.end()) == 8);
        assert(*std::next(c.begin(), 0) == 0);
        assert(*std::next(c.begin(), 1) == 1);
        assert(*std::next(c.begin(), 2) == 4);
        assert(*std::next(c.begin(), 3) == 5);
        assert(*std::next(c.begin(), 4) == 6);
        assert(*std::next(c.begin(), 5) == 7);
        assert(*std::next(c.begin(), 6) == 8);
        assert(*std::next(c.begin(), 7) == 9);

        i = c.erase_after(std::next(c.cbefore_begin(), 2), std::next(c.cbefore_begin(), 3));
        assert(i == std::next(c.begin(), 2));
        assert(std::distance(c.begin(), c.end()) == 8);
        assert(*std::next(c.begin(), 0) == 0);
        assert(*std::next(c.begin(), 1) == 1);
        assert(*std::next(c.begin(), 2) == 4);
        assert(*std::next(c.begin(), 3) == 5);
        assert(*std::next(c.begin(), 4) == 6);
        assert(*std::next(c.begin(), 5) == 7);
        assert(*std::next(c.begin(), 6) == 8);
        assert(*std::next(c.begin(), 7) == 9);

        i = c.erase_after(std::next(c.cbefore_begin(), 5), std::next(c.cbefore_begin(), 9));
        assert(i == c.end());
        assert(std::distance(c.begin(), c.end()) == 5);
        assert(*std::next(c.begin(), 0) == 0);
        assert(*std::next(c.begin(), 1) == 1);
        assert(*std::next(c.begin(), 2) == 4);
        assert(*std::next(c.begin(), 3) == 5);
        assert(*std::next(c.begin(), 4) == 6);

        i = c.erase_after(std::next(c.cbefore_begin(), 0), std::next(c.cbefore_begin(), 2));
        assert(i == c.begin());
        assert(std::distance(c.begin(), c.end()) == 4);
        assert(*std::next(c.begin(), 0) == 1);
        assert(*std::next(c.begin(), 1) == 4);
        assert(*std::next(c.begin(), 2) == 5);
        assert(*std::next(c.begin(), 3) == 6);

        i = c.erase_after(std::next(c.cbefore_begin(), 0), std::next(c.cbefore_begin(), 5));
        assert(i == c.begin());
        assert(i == c.end());
        assert(std::distance(c.begin(), c.end()) == 0);
    }
#endif

  return 0;
}
