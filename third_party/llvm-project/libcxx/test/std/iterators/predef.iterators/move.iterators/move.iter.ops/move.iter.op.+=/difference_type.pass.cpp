//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// requires RandomAccessIterator<Iter>
//   move_iterator& operator+=(difference_type n);
//
//  constexpr in C++17

#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
void
test(It i, typename std::iterator_traits<It>::difference_type n, It x)
{
    std::move_iterator<It> r(i);
    std::move_iterator<It>& rr = r += n;
    assert(r.base() == x);
    assert(&rr == &r);
}

int main(int, char**)
{
    const char* s = "1234567890";
    test(random_access_iterator<const char*>(s+5), 5, random_access_iterator<const char*>(s+10));
    test(s+5, 5, s+10);

#if TEST_STD_VER > 14
    {
    constexpr const char *p = "123456789";
    typedef std::move_iterator<const char *> MI;
    constexpr MI it1 = std::make_move_iterator(p);
    constexpr MI it2 = std::make_move_iterator(p + 5);
    constexpr MI it3 = std::make_move_iterator(p) += 5;
    static_assert(it1 != it2, "");
    static_assert(it1 != it3, "");
    static_assert(it2 == it3, "");
    }
#endif

  return 0;
}
