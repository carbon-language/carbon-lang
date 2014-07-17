//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <experimental/utility>

#include <experimental/utility>

int main()
{
#if _LIBCPP_STD_VER > 11
# ifndef _LIBCPP_UTILITY
#   error "<experimental/utility> must include <utility>"
# endif
#endif /* _LIBCPP_STD_VER > 11 */
}
