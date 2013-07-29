//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// plus

#include <functional>
#include <type_traits>
#include <cassert>

int main()
{
    typedef std::plus<int> F;
    const F f = F();
    static_assert((std::is_base_of<std::binary_function<int, int, int>, F>::value), "");
    assert(f(3, 2) == 5);
#if _LIBCPP_STD_VER > 11
    typedef std::plus<> F2;
    const F2 f2 = F2();
    assert(f2(3,2) == 5);
    assert(f2(3.0, 2) == 5);
    assert(f2(3, 2.5) == 5.5);
#endif
}
