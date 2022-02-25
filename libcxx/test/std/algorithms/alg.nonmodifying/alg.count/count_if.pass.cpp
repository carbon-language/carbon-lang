//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter, Predicate<auto, Iter::value_type> Pred>
//   requires CopyConstructible<Pred>
//   constexpr Iter::difference_type   // constexpr after C++17
//   count_if(Iter first, Iter last, Pred pred);

#include <algorithm>
#include <functional>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

struct eq {
    TEST_CONSTEXPR eq (int val) : v(val) {}
    TEST_CONSTEXPR bool operator () (int v2) const { return v == v2; }
    int v;
    };

#if TEST_STD_VER > 17
TEST_CONSTEXPR bool test_constexpr() {
    int ia[] = {0, 1, 2, 2, 0, 1, 2, 3};
    int ib[] = {1, 2, 3, 4, 5, 6};
    return    (std::count_if(std::begin(ia), std::end(ia), eq(2)) == 3)
           && (std::count_if(std::begin(ib), std::end(ib), eq(9)) == 0)
           ;
    }
#endif

int main(int, char**)
{
    int ia[] = {0, 1, 2, 2, 0, 1, 2, 3};
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);
    assert(std::count_if(cpp17_input_iterator<const int*>(ia),
                         cpp17_input_iterator<const int*>(ia + sa),
                         eq(2)) == 3);
    assert(std::count_if(cpp17_input_iterator<const int*>(ia),
                         cpp17_input_iterator<const int*>(ia + sa),
                         eq(7)) == 0);
    assert(std::count_if(cpp17_input_iterator<const int*>(ia),
                         cpp17_input_iterator<const int*>(ia),
                         eq(2)) == 0);

#if TEST_STD_VER > 17
    static_assert(test_constexpr());
#endif

  return 0;
}
