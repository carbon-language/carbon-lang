//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// #include <initializer_list>

#include <optional>

int main()
{
#if _LIBCPP_STD_VER > 11
    std::initializer_list<int> list;
#endif  // _LIBCPP_STD_VER > 11
}
