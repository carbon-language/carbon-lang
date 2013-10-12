//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test <cstdio>

// gets 

#include <cstdio>

int main()
{
#if _LIBCPP_STD_VER > 11
    (void) std::gets((char *) NULL);
#else
#error
#endif
}
