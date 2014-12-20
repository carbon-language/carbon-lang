//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// T shall be an object type and shall satisfy the requirements of Destructible

#include <experimental/optional>

int main()
{
#if _LIBCPP_STD_VER > 11
    using std::experimental::optional;

    optional<void> opt;
#else
#error
#endif  // _LIBCPP_STD_VER > 11
}
