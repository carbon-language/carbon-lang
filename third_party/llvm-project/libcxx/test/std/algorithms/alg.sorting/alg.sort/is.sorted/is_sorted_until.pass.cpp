//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter>
//   requires LessThanComparable<Iter::value_type>
//   Iter
//   is_sorted_until(Iter first, Iter last);

#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

#if TEST_STD_VER > 17
TEST_CONSTEXPR bool test_constexpr() {
    int ia[] = {0, 1, 0};
    int ib[] = {0, 1, 1};
    return    (std::is_sorted_until(std::begin(ia), std::end(ia)) == ia+2)
           && (std::is_sorted_until(std::begin(ib), std::end(ib)) == ib+3);
    }
#endif

template <class Iter>
void
test()
{
    {
    int a[] = {0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a)) == Iter(a));
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+sa));
    }

    {
    int a[] = {0, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+sa));
    }
    {
    int a[] = {0, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+sa));
    }
    {
    int a[] = {1, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+1));
    }
    {
    int a[] = {1, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+sa));
    }

    {
    int a[] = {0, 0, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+sa));
    }
    {
    int a[] = {0, 0, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+sa));
    }
    {
    int a[] = {0, 1, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+2));
    }
    {
    int a[] = {0, 1, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+sa));
    }
    {
    int a[] = {1, 0, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+1));
    }
    {
    int a[] = {1, 0, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+1));
    }
    {
    int a[] = {1, 1, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+2));
    }
    {
    int a[] = {1, 1, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+sa));
    }

    {
    int a[] = {0, 0, 0, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+sa));
    }
    {
    int a[] = {0, 0, 0, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+sa));
    }
    {
    int a[] = {0, 0, 1, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+3));
    }
    {
    int a[] = {0, 0, 1, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+sa));
    }
    {
    int a[] = {0, 1, 0, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+2));
    }
    {
    int a[] = {0, 1, 0, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+2));
    }
    {
    int a[] = {0, 1, 1, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+3));
    }
    {
    int a[] = {0, 1, 1, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+sa));
    }
    {
    int a[] = {1, 0, 0, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+1));
    }
    {
    int a[] = {1, 0, 0, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+1));
    }
    {
    int a[] = {1, 0, 1, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+1));
    }
    {
    int a[] = {1, 0, 1, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+1));
    }
    {
    int a[] = {1, 1, 0, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+2));
    }
    {
    int a[] = {1, 1, 0, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+2));
    }
    {
    int a[] = {1, 1, 1, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+3));
    }
    {
    int a[] = {1, 1, 1, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(std::is_sorted_until(Iter(a), Iter(a+sa)) == Iter(a+sa));
    }
}

int main(int, char**)
{
    test<forward_iterator<const int*> >();
    test<bidirectional_iterator<const int*> >();
    test<random_access_iterator<const int*> >();
    test<const int*>();

#if TEST_STD_VER > 17
    static_assert(test_constexpr());
#endif

  return 0;
}
