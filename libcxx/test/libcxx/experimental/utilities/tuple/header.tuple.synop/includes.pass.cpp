//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <experimental/tuple>

#include <experimental/tuple>

int main()
{
#if _LIBCPP_STD_VER > 11
# ifndef _LIBCPP_TUPLE
#   error "<experimental/tuple> must include <tuple>"
# endif
#endif /* _LIBCPP_STD_VER > 11 */
}
