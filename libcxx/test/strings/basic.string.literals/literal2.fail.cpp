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
	std::string foo  =   ""s;  // should fail w/conversion operator not found
#else
#error
#endif
}
