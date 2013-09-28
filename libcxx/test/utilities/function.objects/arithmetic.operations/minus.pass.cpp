//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// minus

#include <functional>
#include <type_traits>
#include <cassert>

int main()
{
    typedef std::minus<int> F;
    const F f = F();
    static_assert((std::is_base_of<std::binary_function<int, int, int>, F>::value), "");
    assert(f(3, 2) == 1);
#if _LIBCPP_STD_VER > 11
    typedef std::minus<> F2;
    const F2 f2 = F2();
    assert(f2(3,2) == 1);
    assert(f2(3.0, 2) == 1);
    assert(f2(3, 2.5) == 0.5);

    constexpr int foo = std::minus<int> () (3, 2);
    static_assert ( foo == 1, "" );

    constexpr int bar = std::minus<> () (3.0, 2);
    static_assert ( bar == 1, "" );
#endif
}
