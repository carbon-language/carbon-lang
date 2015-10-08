//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <math.h>

// isinf

// XFAIL: linux

#include <math.h>
#include <type_traits>
#include <cassert>

int main()
{
#ifdef isinf
#error isinf defined
#endif
    static_assert((std::is_same<decltype(isinf((float)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isinf((double)0)), bool>::value), "");
    static_assert((std::is_same<decltype(isinf(0)), bool>::value), "");
    static_assert((std::is_same<decltype(isinf((long double)0)), bool>::value), "");
    assert(isinf(-1.0) == false);
}
