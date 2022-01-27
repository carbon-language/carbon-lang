//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <numeric>

// Became constexpr in C++20
// template <InputIterator Iter1, InputIterator Iter2, MoveConstructible T>
//   requires HasMultiply<Iter1::reference, Iter2::reference>
//         && HasPlus<T, HasMultiply<Iter1::reference, Iter2::reference>::result_type>
//         && HasAssign<T,
//                      HasPlus<T,
//                              HasMultiply<Iter1::reference,
//                                          Iter2::reference>::result_type>::result_type>
//   T
//   inner_product(Iter1 first1, Iter1 last1, Iter2 first2, T init);

#include <numeric>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class Iter1, class Iter2, class T>
TEST_CONSTEXPR_CXX20 void
test(Iter1 first1, Iter1 last1, Iter2 first2, T init, T x)
{
    assert(std::inner_product(first1, last1, first2, init) == x);
}

template <class Iter1, class Iter2>
TEST_CONSTEXPR_CXX20 void
test()
{
    int a[] = {1, 2, 3, 4, 5, 6};
    int b[] = {6, 5, 4, 3, 2, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    test(Iter1(a), Iter1(a), Iter2(b), 0, 0);
    test(Iter1(a), Iter1(a), Iter2(b), 10, 10);
    test(Iter1(a), Iter1(a+1), Iter2(b), 0, 6);
    test(Iter1(a), Iter1(a+1), Iter2(b), 10, 16);
    test(Iter1(a), Iter1(a+2), Iter2(b), 0, 16);
    test(Iter1(a), Iter1(a+2), Iter2(b), 10, 26);
    test(Iter1(a), Iter1(a+sa), Iter2(b), 0, 56);
    test(Iter1(a), Iter1(a+sa), Iter2(b), 10, 66);
}

TEST_CONSTEXPR_CXX20 bool
test()
{
    test<cpp17_input_iterator<const int*>, cpp17_input_iterator<const int*> >();
    test<cpp17_input_iterator<const int*>, forward_iterator<const int*> >();
    test<cpp17_input_iterator<const int*>, bidirectional_iterator<const int*> >();
    test<cpp17_input_iterator<const int*>, random_access_iterator<const int*> >();
    test<cpp17_input_iterator<const int*>, const int*>();

    test<forward_iterator<const int*>, cpp17_input_iterator<const int*> >();
    test<forward_iterator<const int*>, forward_iterator<const int*> >();
    test<forward_iterator<const int*>, bidirectional_iterator<const int*> >();
    test<forward_iterator<const int*>, random_access_iterator<const int*> >();
    test<forward_iterator<const int*>, const int*>();

    test<bidirectional_iterator<const int*>, cpp17_input_iterator<const int*> >();
    test<bidirectional_iterator<const int*>, forward_iterator<const int*> >();
    test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*> >();
    test<bidirectional_iterator<const int*>, random_access_iterator<const int*> >();
    test<bidirectional_iterator<const int*>, const int*>();

    test<random_access_iterator<const int*>, cpp17_input_iterator<const int*> >();
    test<random_access_iterator<const int*>, forward_iterator<const int*> >();
    test<random_access_iterator<const int*>, bidirectional_iterator<const int*> >();
    test<random_access_iterator<const int*>, random_access_iterator<const int*> >();
    test<random_access_iterator<const int*>, const int*>();

    test<const int*, cpp17_input_iterator<const int*> >();
    test<const int*, forward_iterator<const int*> >();
    test<const int*, bidirectional_iterator<const int*> >();
    test<const int*, random_access_iterator<const int*> >();
    test<const int*, const int*>();

    return true;
}

int main(int, char**)
{
    test();
#if TEST_STD_VER > 17
    static_assert(test());
#endif
    return 0;
}
