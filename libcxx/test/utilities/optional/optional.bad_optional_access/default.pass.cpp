//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// class bad_optional_access is not default constructible

#include <optional>
#include <type_traits>

#if _LIBCPP_STD_VER > 11

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    static_assert(!std::is_default_constructible<std::bad_optional_access>::value, "");
#endif  // _LIBCPP_STD_VER > 11
}
