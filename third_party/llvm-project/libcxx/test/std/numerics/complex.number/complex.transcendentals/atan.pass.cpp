//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-AIX-FIXME

// <complex>

// template<class T>
//   complex<T>
//   atan(const complex<T>& x);

#include <complex>
#include <cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
void
test(const std::complex<T>& c, std::complex<T> x)
{
    assert(atan(c) == x);
}

template <class T>
void
test()
{
    test(std::complex<T>(0, 0), std::complex<T>(0, 0));
}

void test_edges()
{
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        std::complex<double> r = atan(testcases[i]);
        std::complex<double> t1(-imag(testcases[i]), real(testcases[i]));
        std::complex<double> t2 = atanh(t1);
        std::complex<double> z(imag(t2), -real(t2));
        if (std::isnan(real(r)))
            assert(std::isnan(real(z)));
        else
        {
            assert(real(r) == real(z));
            assert(std::signbit(real(r)) == std::signbit(real(z)));
        }
        if (std::isnan(imag(r)))
            assert(std::isnan(imag(z)));
        else
        {
            assert(imag(r) == imag(z));
            assert(std::signbit(imag(r)) == std::signbit(imag(z)));
        }
    }
}

int main(int, char**)
{
    test<float>();
    test<double>();
    test<long double>();
    test_edges();

  return 0;
}
