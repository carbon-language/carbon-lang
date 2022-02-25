//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter, EquivalenceRelation<auto, Iter::value_type> Pred>
//   requires CopyConstructible<Pred>
//   constexpr Iter  // constexpr after C++17
//   adjacent_find(Iter first, Iter last, Pred pred);

#include <algorithm>
#include <functional>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"


#if TEST_STD_VER > 17
TEST_CONSTEXPR bool eq (int a, int b) { return a == b; }

TEST_CONSTEXPR bool test_constexpr() {
    int ia[] = {0, 1, 2, 2, 0, 1, 2, 3};
    int ib[] = {0, 1, 2, 7, 0, 1, 2, 3};

    return  (std::adjacent_find(std::begin(ia), std::end(ia), eq) == ia+2)
         && (std::adjacent_find(std::begin(ib), std::end(ib), eq) == std::end(ib))
         ;
    }
#endif

int main(int, char**)
{
    int ia[] = {0, 1, 2, 2, 0, 1, 2, 3};
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);
    assert(std::adjacent_find(forward_iterator<const int*>(ia),
                              forward_iterator<const int*>(ia + sa),
                              std::equal_to<int>()) ==
                              forward_iterator<const int*>(ia+2));
    assert(std::adjacent_find(forward_iterator<const int*>(ia),
                              forward_iterator<const int*>(ia),
                              std::equal_to<int>()) ==
                              forward_iterator<const int*>(ia));
    assert(std::adjacent_find(forward_iterator<const int*>(ia+3),
                              forward_iterator<const int*>(ia + sa),
                              std::equal_to<int>()) ==
                              forward_iterator<const int*>(ia+sa));

#if TEST_STD_VER > 17
    static_assert(test_constexpr());
#endif

  return 0;
}
