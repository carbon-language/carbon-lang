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
    using std::experimental::erased_type;
    constexpr erased_type e{};
#endif
}
