//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <numeric>

// Became constexpr in C++20
// template <InputIterator InIter, OutputIterator<auto, const InIter::value_type&> OutIter>
//   requires HasPlus<InIter::value_type, InIter::reference>
//         && HasAssign<InIter::value_type,
//                      HasPlus<InIter::value_type, InIter::reference>::result_type>
//         && Constructible<InIter::value_type, InIter::reference>
//   OutIter
//   partial_sum(InIter first, InIter last, OutIter result);

#include <numeric>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class InIter, class OutIter>
TEST_CONSTEXPR_CXX20 void
test()
{
    int ia[] = {1, 2, 3, 4, 5};
    int ir[] = {1, 3, 6, 10, 15};
    const unsigned s = sizeof(ia) / sizeof(ia[0]);
    int ib[s] = {0};
    OutIter r = std::partial_sum(InIter(ia), InIter(ia+s), OutIter(ib));
    assert(base(r) == ib + s);
    for (unsigned i = 0; i < s; ++i)
        assert(ib[i] == ir[i]);
}

TEST_CONSTEXPR_CXX20 bool
test()
{
    test<cpp17_input_iterator<const int*>, output_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, forward_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, bidirectional_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, random_access_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, int*>();

    test<forward_iterator<const int*>, output_iterator<int*> >();
    test<forward_iterator<const int*>, forward_iterator<int*> >();
    test<forward_iterator<const int*>, bidirectional_iterator<int*> >();
    test<forward_iterator<const int*>, random_access_iterator<int*> >();
    test<forward_iterator<const int*>, int*>();

    test<bidirectional_iterator<const int*>, output_iterator<int*> >();
    test<bidirectional_iterator<const int*>, forward_iterator<int*> >();
    test<bidirectional_iterator<const int*>, bidirectional_iterator<int*> >();
    test<bidirectional_iterator<const int*>, random_access_iterator<int*> >();
    test<bidirectional_iterator<const int*>, int*>();

    test<random_access_iterator<const int*>, output_iterator<int*> >();
    test<random_access_iterator<const int*>, forward_iterator<int*> >();
    test<random_access_iterator<const int*>, bidirectional_iterator<int*> >();
    test<random_access_iterator<const int*>, random_access_iterator<int*> >();
    test<random_access_iterator<const int*>, int*>();

    test<const int*, output_iterator<int*> >();
    test<const int*, forward_iterator<int*> >();
    test<const int*, bidirectional_iterator<int*> >();
    test<const int*, random_access_iterator<int*> >();
    test<const int*, int*>();

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
