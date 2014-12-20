//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// class bad_optional_access is default constructible

#include <experimental/optional>
#include <type_traits>

int main()
{
#if _LIBCPP_STD_VER > 11
    using std::experimental::bad_optional_access;
    bad_optional_access ex;
#endif  // _LIBCPP_STD_VER > 11
}
