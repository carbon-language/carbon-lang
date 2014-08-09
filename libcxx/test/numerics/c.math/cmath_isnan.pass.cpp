//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <cmath>

// isinf

// XFAIL: linux

#include <cmath>
#include <type_traits>
#include <cassert>

int main()
{
#ifdef isnan
#error isnan defined
#endif
    static_assert((std::is_same<decltype(std::isnan((float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::isnan((double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::isnan(0)), bool>::value), "");
    static_assert((std::is_same<decltype(std::isnan((long double)0)), bool>::value), "");
    assert(std::isnan(-1.0) == false);
}
