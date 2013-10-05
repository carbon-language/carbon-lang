// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <string>
#include <cassert>

int main()
{
#if _LIBCPP_STD_VER > 11 
    using namespace std;

    string foo  =   ""s;
#endif
}
