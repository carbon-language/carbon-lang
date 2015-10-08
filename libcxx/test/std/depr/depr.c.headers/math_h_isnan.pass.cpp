//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <math.h>

// isnan

// XFAIL: linux

#include <math.h>
#include <type_traits>
#include <cassert>

int main()
{
#ifdef isnan
#error isnan defined
#endif
    static_assert((std::is_same<decltype(isnan((float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isnan((double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isnan(0)), bool>::value), "");
    static_assert((std::is_same<decltype(isnan((long double)0)), bool>::value), "");
    assert(isnan(-1.0) == false);
}
