//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <complex>

// template<> class complex<double>
// { 
// public: 
//     constexpr complex(const complex<float>&); 
// };

#include <complex>
#include <cassert>

int main()
{
    const std::complex<float> cd(2.5, 3.5);
    std::complex<double> cf(cd);
    assert(cf.real() == cd.real());
    assert(cf.imag() == cd.imag());
}
