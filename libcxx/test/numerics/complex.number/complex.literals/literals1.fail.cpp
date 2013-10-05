// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <complex>
#include <cassert>

int main()
{
#if _LIBCPP_STD_VER > 11 
    std::complex<float> foo  = 1.0if;  // should fail w/conversion operator not found
#else
#error
#endif
}
