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

#include <experimental/optional>
#include <string>
#include <cstring>
#include <type_traits>
#include <cassert>

int main()
{
#if _LIBCPP_STD_VER > 11
    using std::experimental::bad_optional_access;

    static_assert(std::is_nothrow_copy_constructible<bad_optional_access>::value, "");
    const std::string s("another message");
    bad_optional_access e1(s);
    bad_optional_access e2 = e1;
    assert(std::strcmp(e1.what(), e2.what()) == 0);
#endif  // _LIBCPP_STD_VER > 11
}
