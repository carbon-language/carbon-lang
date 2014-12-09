//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// A program that necessitates the instantiation of template optional for a
// reference type is ill-formed.

#include <experimental/optional>

int main()
{
#if _LIBCPP_STD_VER > 11
    using std::experimental::optional;

    optional<int&> opt;
#else
#error
#endif  // _LIBCPP_STD_VER > 11
}
