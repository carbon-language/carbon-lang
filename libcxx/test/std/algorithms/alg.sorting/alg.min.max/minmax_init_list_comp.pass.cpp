//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <algorithm>

// template<class T, class Compare>
//   pair<T, T>
//   minmax(initializer_list<T> t, Compare comp);
//
//  Complexity: At most (3/2) * t.size() applications of the corresponding predicate.

#include <algorithm>
#include <functional>
#include <cassert>

#include "test_macros.h"
#include "counting_predicates.h"

bool all_equal(int, int) { return false; } // everything is equal

void test_all_equal(std::initializer_list<int> il)
{
    binary_counting_predicate<bool(*)(int, int), int, int> pred (all_equal);
    std::pair<int, int> p = std::minmax(il, std::ref(pred));
    const int *ptr = il.end();
    assert(p.first == *il.begin());
    assert(p.second == *--ptr);
    assert(pred.count() <= ((3 * il.size()) / 2));
}

int main(int, char**)
{
    assert((std::minmax({1, 2, 3}, std::greater<int>()) == std::pair<int, int>(3, 1)));
    assert((std::minmax({1, 3, 2}, std::greater<int>()) == std::pair<int, int>(3, 1)));
    assert((std::minmax({2, 1, 3}, std::greater<int>()) == std::pair<int, int>(3, 1)));
    assert((std::minmax({2, 3, 1}, std::greater<int>()) == std::pair<int, int>(3, 1)));
    assert((std::minmax({3, 1, 2}, std::greater<int>()) == std::pair<int, int>(3, 1)));
    assert((std::minmax({3, 2, 1}, std::greater<int>()) == std::pair<int, int>(3, 1)));
    assert((std::minmax({1, 2, 3}, all_equal          ) == std::pair<int, int>(1, 3)));

    binary_counting_predicate<std::greater<int>, int, int> pred ((std::greater<int>()));
    assert((std::minmax({1, 2, 2, 3, 3, 3, 5, 5, 5, 5, 5, 3}, std::ref(pred)) == std::pair<int, int>(5, 1)));
    assert(pred.count() <= 18); // size == 12

    test_all_equal({0});
    test_all_equal({0,1});
    test_all_equal({0,1,2});
    test_all_equal({0,1,2,3});
    test_all_equal({0,1,2,3,4});
    test_all_equal({0,1,2,3,4,5});
    test_all_equal({0,1,2,3,4,5,6});
    test_all_equal({0,1,2,3,4,5,6,7});
    test_all_equal({0,1,2,3,4,5,6,7,8});
    test_all_equal({0,1,2,3,4,5,6,7,8,9});
    test_all_equal({0,1,2,3,4,5,6,7,8,9,10});
    test_all_equal({0,1,2,3,4,5,6,7,8,9,10,11});

#if TEST_STD_VER >= 14
    {
    static_assert((std::minmax({1, 2, 3}, std::greater<int>()) == std::pair<int, int>(3, 1)), "");
    static_assert((std::minmax({1, 3, 2}, std::greater<int>()) == std::pair<int, int>(3, 1)), "");
    static_assert((std::minmax({2, 1, 3}, std::greater<int>()) == std::pair<int, int>(3, 1)), "");
    static_assert((std::minmax({2, 3, 1}, std::greater<int>()) == std::pair<int, int>(3, 1)), "");
    static_assert((std::minmax({3, 1, 2}, std::greater<int>()) == std::pair<int, int>(3, 1)), "");
    static_assert((std::minmax({3, 2, 1}, std::greater<int>()) == std::pair<int, int>(3, 1)), "");
    }
#endif

  return 0;
}
