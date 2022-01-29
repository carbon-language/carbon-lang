//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// move_iterator operator--(int);
//
//  constexpr in C++17

#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
void
test(It i, It x)
{
    std::move_iterator<It> r(i);
    std::move_iterator<It> rr = r--;
    assert(r.base() == x);
    assert(rr.base() == i);
}

int main(int, char**)
{
    char s[] = "123";
    test(bidirectional_iterator<char*>(s+1), bidirectional_iterator<char*>(s));
    test(random_access_iterator<char*>(s+1), random_access_iterator<char*>(s));
    test(s+1, s);

#if TEST_STD_VER > 14
    {
    constexpr const char *p = "123456789";
    typedef std::move_iterator<const char *> MI;
    constexpr MI it1 = std::make_move_iterator(p);
    constexpr MI it2 = std::make_move_iterator(p+1);
    static_assert(it1 != it2, "");
    constexpr MI it3 = std::make_move_iterator(p+1) --;
    static_assert(it1 != it3, "");
    static_assert(it2 == it3, "");
    }
#endif

  return 0;
}
