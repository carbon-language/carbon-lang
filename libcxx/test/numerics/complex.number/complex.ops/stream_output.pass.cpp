//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T, class charT, class traits> 
//   basic_ostream<charT, traits>& 
//   operator<<(basic_ostream<charT, traits>& o, const complex<T>& x);

#include <complex>
#include <sstream>
#include <cassert>

int main()
{
    std::complex<double> c(1, 2);
    std::ostringstream os;
    os << c;
    assert(os.str() == "(1,2)");
}
