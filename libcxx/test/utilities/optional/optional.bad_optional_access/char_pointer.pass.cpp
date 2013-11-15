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
// explicit bad_optional_access(const char* what_arg);

#include <experimental/optional>
#include <string>
#include <cstring>
#include <cassert>

int main()
{
#if _LIBCPP_STD_VER > 11
    using std::experimental::bad_optional_access;

    const char* s = "message";
    bad_optional_access e(s);
    assert(std::strcmp(e.what(), s) == 0);
#endif  // _LIBCPP_STD_VER > 11
}
