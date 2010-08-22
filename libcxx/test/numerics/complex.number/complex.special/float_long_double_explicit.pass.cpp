//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <complex>

// template<> class complex<float>
// {
// public:
//     explicit constexpr complex(const complex<long double>&);
// };

#include <complex>
#include <cassert>

int main()
{
    const std::complex<long double> cd(2.5, 3.5);
    std::complex<float> cf(cd);
    assert(cf.real() == cd.real());
    assert(cf.imag() == cd.imag());
}
