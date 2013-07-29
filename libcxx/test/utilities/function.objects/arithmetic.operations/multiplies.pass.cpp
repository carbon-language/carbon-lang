//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// multiplies

#include <functional>
#include <type_traits>
#include <cassert>

int main()
{
    typedef std::multiplies<int> F;
    const F f = F();
    static_assert((std::is_base_of<std::binary_function<int, int, int>, F>::value), "");
    assert(f(3, 2) == 6);
#if _LIBCPP_STD_VER > 11
    typedef std::multiplies<> F2;
    const F2 f2 = F2();
    assert(f2(3,2) == 6);
    assert(f2(3.0, 2) == 6);
    assert(f2(3, 2.5) == 7.5); // exact in binary
#endif
}
