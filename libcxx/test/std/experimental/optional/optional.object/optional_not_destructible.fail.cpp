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

#if _LIBCPP_STD_VER > 11

using std::experimental::optional;

struct X
{
private:
    ~X() {}
};

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    optional<X> opt;
#else
#error
#endif  // _LIBCPP_STD_VER > 11
}
