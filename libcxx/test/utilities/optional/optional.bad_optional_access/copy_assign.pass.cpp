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
// bad_optional_access& operator=(const bad_optional_access&);

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
    static_assert(std::is_nothrow_copy_assignable<std::bad_optional_access>::value, "");
    const std::string s1("one message");
    const std::string s2("another message");
    std::bad_optional_access e1(s1);
    std::bad_optional_access e2(s2);
    assert(std::strcmp(e1.what(), e2.what()) != 0);
    e1 = e2;
    assert(std::strcmp(e1.what(), e2.what()) == 0);
#endif  // _LIBCPP_STD_VER > 11
}
