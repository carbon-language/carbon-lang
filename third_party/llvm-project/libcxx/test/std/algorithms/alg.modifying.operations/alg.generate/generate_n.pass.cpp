//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class Iter, IntegralLike Size, Callable Generator>
//   requires OutputIterator<Iter, Generator::result_type>
//         && CopyConstructible<Generator>
//   constexpr void      // constexpr after c++17
//   generate_n(Iter first, Size n, Generator gen);

#include <algorithm>
#include <cassert>

#include "test_iterators.h"
#include "test_macros.h"
#include "user_defined_integral.h"

TEST_MSVC_DIAGNOSTIC_IGNORED(4244) // conversion from 'const double' to 'int', possible loss of data

struct gen_test
{
    TEST_CONSTEXPR int operator()() const {return 2;}
};


#if TEST_STD_VER > 17
TEST_CONSTEXPR bool test_constexpr() {
    const size_t N = 5;
    int ib[] = {0, 0, 0, 0, 0, 0}; // one bigger than N

    auto it = std::generate_n(std::begin(ib), N, gen_test());

    return it == (std::begin(ib) + N)
        && std::all_of(std::begin(ib), it, [](int x) { return x == 2; })
        && *it == 0 // don't overwrite the last value in the output array
        ;
    }
#endif


template <class Iter, class Size>
void
test2()
{
    const unsigned n = 4;
    int ia[n] = {0};
    assert(std::generate_n(Iter(ia), Size(n), gen_test()) == Iter(ia+n));
    assert(ia[0] == 2);
    assert(ia[1] == 2);
    assert(ia[2] == 2);
    assert(ia[3] == 2);
}

template <class Iter>
void
test()
{
    test2<Iter, int>();
    test2<Iter, unsigned int>();
    test2<Iter, long>();
    test2<Iter, unsigned long>();
    test2<Iter, UserDefinedIntegral<unsigned> >();
    test2<Iter, float>();
    test2<Iter, double>();  // this is PR#35498
    test2<Iter, long double>();
}

int main(int, char**)
{
    test<forward_iterator<int*> >();
    test<bidirectional_iterator<int*> >();
    test<random_access_iterator<int*> >();
    test<int*>();

#if TEST_STD_VER > 17
    static_assert(test_constexpr());
#endif

  return 0;
}
