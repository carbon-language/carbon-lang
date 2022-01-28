//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<Arithmetic T>
//   T
//   imag(const T& x);

#include <complex>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T, int x>
void
test(typename std::enable_if<std::is_integral<T>::value>::type* = 0)
{
    static_assert((std::is_same<decltype(std::imag(T(x))), double>::value), "");
    assert(std::imag(x) == 0);
#if TEST_STD_VER > 11
    constexpr T val {x};
    static_assert(std::imag(val) == 0, "");
    constexpr std::complex<T> t{val, val};
    static_assert(t.imag() == x, "" );
#endif
}

template <class T, int x>
void
test(typename std::enable_if<!std::is_integral<T>::value>::type* = 0)
{
    static_assert((std::is_same<decltype(std::imag(T(x))), T>::value), "");
    assert(std::imag(x) == 0);
#if TEST_STD_VER > 11
    constexpr T val {x};
    static_assert(std::imag(val) == 0, "");
    constexpr std::complex<T> t{val, val};
    static_assert(t.imag() == x, "" );
#endif
}

template <class T>
void
test()
{
    test<T, 0>();
    test<T, 1>();
    test<T, 10>();
}

int main(int, char**)
{
    test<float>();
    test<double>();
    test<long double>();
    test<int>();
    test<unsigned>();
    test<long long>();

  return 0;
}
