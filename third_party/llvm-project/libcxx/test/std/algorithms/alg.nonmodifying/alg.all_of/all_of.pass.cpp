//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template <class InputIterator, class Predicate>
//     constexpr bool       // constexpr after C++17
//   all_of(InputIterator first, InputIterator last, Predicate pred);

#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

struct test1
{
    TEST_CONSTEXPR bool operator()(const int& i) const
    {
        return i % 2 == 0;
    }
};

#if TEST_STD_VER > 17
TEST_CONSTEXPR bool test_constexpr() {
    int ia[] = {2, 4, 6, 8};
    int ib[] = {2, 4, 5, 8};
    return  std::all_of(std::begin(ia), std::end(ia), test1())
        && !std::all_of(std::begin(ib), std::end(ib), test1())
        ;
    }
#endif

int main(int, char**)
{
    {
        int ia[] = {2, 4, 6, 8};
        const unsigned sa = sizeof(ia)/sizeof(ia[0]);
        assert(std::all_of(cpp17_input_iterator<const int*>(ia),
                           cpp17_input_iterator<const int*>(ia + sa), test1()) == true);
        assert(std::all_of(cpp17_input_iterator<const int*>(ia),
                           cpp17_input_iterator<const int*>(ia), test1()) == true);
    }
    {
        const int ia[] = {2, 4, 5, 8};
        const unsigned sa = sizeof(ia)/sizeof(ia[0]);
        assert(std::all_of(cpp17_input_iterator<const int*>(ia),
                           cpp17_input_iterator<const int*>(ia + sa), test1()) == false);
        assert(std::all_of(cpp17_input_iterator<const int*>(ia),
                           cpp17_input_iterator<const int*>(ia), test1()) == true);
    }

#if TEST_STD_VER > 17
    static_assert(test_constexpr());
#endif

  return 0;
}
