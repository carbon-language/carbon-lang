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
//   acos(const complex<T>& x);

#include <complex>
#include <cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
void
test(const std::complex<T>& c, std::complex<T> x)
{
    assert(acos(c) == x);
}

template <class T>
void
test()
{
    test(std::complex<T>(INFINITY, 1), std::complex<T>(0, -INFINITY));
}

void test_edges()
{
    const double pi = std::atan2(+0., -0.);
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        std::complex<double> r = acos(testcases[i]);
        if (testcases[i].real() == 0 && testcases[i].imag() == 0)
        {
            is_about(r.real(), pi/2);
            assert(r.imag() == 0);
            assert(std::signbit(testcases[i].imag()) != std::signbit(r.imag()));
        }
        else if (testcases[i].real() == 0 && std::isnan(testcases[i].imag()))
        {
            is_about(r.real(), pi/2);
            assert(std::isnan(r.imag()));
        }
        else if (std::isfinite(testcases[i].real()) && std::isinf(testcases[i].imag()))
        {
            is_about(r.real(), pi/2);
            assert(std::isinf(r.imag()));
            assert(std::signbit(testcases[i].imag()) != std::signbit(r.imag()));
        }
        else if (std::isfinite(testcases[i].real()) && testcases[i].real() != 0 && std::isnan(testcases[i].imag()))
        {
            assert(std::isnan(r.real()));
            assert(std::isnan(r.imag()));
        }
        else if (std::isinf(testcases[i].real()) && testcases[i].real() < 0 && std::isfinite(testcases[i].imag()))
        {
            is_about(r.real(), pi);
            assert(std::isinf(r.imag()));
            assert(std::signbit(testcases[i].imag()) != std::signbit(r.imag()));
        }
        else if (std::isinf(testcases[i].real()) && testcases[i].real() > 0 && std::isfinite(testcases[i].imag()))
        {
            assert(r.real() == 0);
            assert(!std::signbit(r.real()));
            assert(std::isinf(r.imag()));
            assert(std::signbit(testcases[i].imag()) != std::signbit(r.imag()));
        }
        else if (std::isinf(testcases[i].real()) && testcases[i].real() < 0 && std::isinf(testcases[i].imag()))
        {
            is_about(r.real(), 0.75 * pi);
            assert(std::isinf(r.imag()));
            assert(std::signbit(testcases[i].imag()) != std::signbit(r.imag()));
        }
        else if (std::isinf(testcases[i].real()) && testcases[i].real() > 0 && std::isinf(testcases[i].imag()))
        {
            is_about(r.real(), 0.25 * pi);
            assert(std::isinf(r.imag()));
            assert(std::signbit(testcases[i].imag()) != std::signbit(r.imag()));
        }
        else if (std::isinf(testcases[i].real()) && std::isnan(testcases[i].imag()))
        {
            assert(std::isnan(r.real()));
            assert(std::isinf(r.imag()));
        }
        else if (std::isnan(testcases[i].real()) && std::isfinite(testcases[i].imag()))
        {
            assert(std::isnan(r.real()));
            assert(std::isnan(r.imag()));
        }
        else if (std::isnan(testcases[i].real()) && std::isinf(testcases[i].imag()))
        {
            assert(std::isnan(r.real()));
            assert(std::isinf(r.imag()));
            assert(std::signbit(testcases[i].imag()) != std::signbit(r.imag()));
        }
        else if (std::isnan(testcases[i].real()) && std::isnan(testcases[i].imag()))
        {
            assert(std::isnan(r.real()));
            assert(std::isnan(r.imag()));
        }
        else if (!std::signbit(testcases[i].real()) && !std::signbit(testcases[i].imag()))
        {
            assert(!std::signbit(r.real()));
            assert( std::signbit(r.imag()));
        }
        else if (std::signbit(testcases[i].real()) && !std::signbit(testcases[i].imag()))
        {
            assert(!std::signbit(r.real()));
            assert( std::signbit(r.imag()));
        }
        else if (std::signbit(testcases[i].real()) && std::signbit(testcases[i].imag()))
        {
            assert(!std::signbit(r.real()));
            assert(!std::signbit(r.imag()));
        }
        else if (!std::signbit(testcases[i].real()) && std::signbit(testcases[i].imag()))
        {
            assert(!std::signbit(r.real()));
            assert(!std::signbit(r.imag()));
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
