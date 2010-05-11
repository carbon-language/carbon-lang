//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <complex>

// complex& operator= (const T&);

#include <complex>
#include <cassert>

template <class T>
void
test()
{
    std::complex<T> c;
    assert(c.real() == 0);
    assert(c.imag() == 0);
    c = 1.5;
    assert(c.real() == 1.5);
    assert(c.imag() == 0);
    c = -1.5;
    assert(c.real() == -1.5);
    assert(c.imag() == 0);
}

int main()
{
    test<float>();
    test<double>();
    test<long double>();
}
