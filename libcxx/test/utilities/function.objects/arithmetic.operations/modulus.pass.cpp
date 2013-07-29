//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// modulus

#include <functional>
#include <type_traits>
#include <cassert>

int main()
{
    typedef std::modulus<int> F;
    const F f = F();
    static_assert((std::is_base_of<std::binary_function<int, int, int>, F>::value), "");
    assert(f(36, 8) == 4);
#if _LIBCPP_STD_VER > 11
    typedef std::modulus<> F2;
    const F2 f2 = F2();
    assert(f2(36, 8) == 4);
    assert(f2(36L, 8) == 4);
    assert(f2(36, 8L) == 4);
#endif
}
