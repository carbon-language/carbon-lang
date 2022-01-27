//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T>
//   complex<T>
//   proj(const complex<T>& x);

#include <complex>
#include <cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
void
test(const std::complex<T>& z, std::complex<T> x)
{
    assert(proj(z) == x);
}

template <class T>
void
test()
{
    test(std::complex<T>(1, 2), std::complex<T>(1, 2));
    test(std::complex<T>(-1, 2), std::complex<T>(-1, 2));
    test(std::complex<T>(1, -2), std::complex<T>(1, -2));
    test(std::complex<T>(-1, -2), std::complex<T>(-1, -2));
}

void test_edges()
{
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        std::complex<double> r = proj(testcases[i]);
        switch (classify(testcases[i]))
        {
        case zero:
        case non_zero:
            assert(r == testcases[i]);
            assert(std::signbit(real(r)) == std::signbit(real(testcases[i])));
            assert(std::signbit(imag(r)) == std::signbit(imag(testcases[i])));
            break;
        case inf:
            assert(std::isinf(real(r)) && real(r) > 0);
            assert(imag(r) == 0);
            assert(std::signbit(imag(r)) == std::signbit(imag(testcases[i])));
            break;
        case NaN:
        case non_zero_nan:
            assert(classify(r) == classify(testcases[i]));
            break;
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
