//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T>
//   T
//   real(const complex<T>& x);

#include <complex>
#include <cassert>

#include "test_macros.h"

template <class T>
void
test()
{
    std::complex<T> z(1.5, 2.5);
    assert(real(z) == 1.5);
}

int main(int, char**)
{
    test<float>();
    test<double>();
    test<long double>();

  return 0;
}
