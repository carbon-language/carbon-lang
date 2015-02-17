//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <experimental/ratio>

// Test that <ratio> is included.

#include <experimental/ratio>

#ifndef _LIBCPP_STD_VER > 11
# ifndef _LIBCPP_RATIO
#   error " <experimental/ratio> must include <ratio>"
# endif
#endif

int main()
{
}
