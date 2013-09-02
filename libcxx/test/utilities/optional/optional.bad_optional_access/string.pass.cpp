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
// explicit bad_optional_access(const string& what_arg);

#include <optional>
#include <string>
#include <cstring>
#include <cassert>

#if _LIBCPP_STD_VER > 11

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    const std::string s("message");
    std::bad_optional_access e(s);
    assert(std::strcmp(e.what(), s.c_str()) == 0);
#endif  // _LIBCPP_STD_VER > 11
}
