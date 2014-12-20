//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// enable_if

#include <type_traits>

int main()
{
#if _LIBCPP_STD_VER > 11
    typedef std::enable_if_t<false> A;
#else
    static_assert ( false, "" );
#endif
}
