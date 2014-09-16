//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter1, InputIterator Iter2>
//   requires HasEqualTo<Iter1::value_type, Iter2::value_type>
//   pair<Iter1, Iter2>
//   mismatch(Iter1 first1, Iter1 last1, Iter2 first2);

#include <algorithm>
#include <cassert>

#include "test_iterators.h"

#if _LIBCPP_STD_VER > 11
#define HAS_FOUR_ITERATOR_VERSION
#endif

int main()
{
    int ia[] = {0, 1, 2, 2, 0, 1, 2, 3};
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);
    int ib[] = {0, 1, 2, 3, 0, 1, 2, 3};
    assert(std::mismatch(input_iterator<const int*>(ia),
                         input_iterator<const int*>(ia + sa),
                         input_iterator<const int*>(ib)) ==
                         (std::pair<input_iterator<const int*>,
                                    input_iterator<const int*> >(
                            input_iterator<const int*>(ia+3),
                            input_iterator<const int*>(ib+3))));

    assert(std::mismatch(comma_iterator<const int*>(ia),
                         comma_iterator<const int*>(ia + sa),
                         comma_iterator<const int*>(ib)) ==
                         (std::pair<comma_iterator<const int*>,
                                    comma_iterator<const int*> >(
                            comma_iterator<const int*>(ia+3),
                            comma_iterator<const int*>(ib+3))));

#ifdef HAS_FOUR_ITERATOR_VERSION
    assert(std::mismatch(input_iterator<const int*>(ia),
                         input_iterator<const int*>(ia + sa),
                         input_iterator<const int*>(ib),
                         input_iterator<const int*>(ib + sa)) ==
                         (std::pair<input_iterator<const int*>,
                                    input_iterator<const int*> >(
                            input_iterator<const int*>(ia+3),
                            input_iterator<const int*>(ib+3))));

    assert(std::mismatch(input_iterator<const int*>(ia),
                         input_iterator<const int*>(ia + sa),
                         input_iterator<const int*>(ib),
                         input_iterator<const int*>(ib + 2)) ==
                         (std::pair<input_iterator<const int*>,
                                    input_iterator<const int*> >(
                            input_iterator<const int*>(ia+2),
                            input_iterator<const int*>(ib+2))));
#endif
}
