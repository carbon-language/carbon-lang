//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// class bad_optional_access;
// bad_optional_access(const bad_optional_access&);

#include <optional>
#include <string>
#include <cstring>
#include <type_traits>
#include <cassert>

#if _LIBCPP_STD_VER > 11

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    static_assert(std::is_nothrow_copy_constructible<std::bad_optional_access>::value, "");
    const std::string s("another message");
    std::bad_optional_access e1(s);
    std::bad_optional_access e2 = e1;
    assert(std::strcmp(e1.what(), e2.what()) == 0);
#endif  // _LIBCPP_STD_VER > 11
}
