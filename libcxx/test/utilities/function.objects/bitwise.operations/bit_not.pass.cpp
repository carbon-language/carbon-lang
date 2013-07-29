//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// bit_not

#include <functional>
#include <type_traits>
#include <cassert>

int main()
{
#if _LIBCPP_STD_VER > 11
    typedef std::bit_not<int> F;
    const F f = F();
    static_assert((std::is_base_of<std::unary_function<int, int>, F>::value), "");
    assert((f(0xEA95) & 0xFFFF ) == 0x156A);
    assert((f(0x58D3) & 0xFFFF ) == 0xA72C);
    assert((f(0)      & 0xFFFF ) == 0xFFFF);
    assert((f(0xFFFF) & 0xFFFF ) == 0);

    typedef std::bit_not<> F2;
    const F2 f2 = F2();
    assert((f2(0xEA95)  & 0xFFFF ) == 0x156A);
    assert((f2(0xEA95L) & 0xFFFF ) == 0x156A);
    assert((f2(0x58D3)  & 0xFFFF ) == 0xA72C);
    assert((f2(0x58D3L) & 0xFFFF ) == 0xA72C);
    assert((f2(0)       & 0xFFFF ) == 0xFFFF);
    assert((f2(0L)      & 0xFFFF ) == 0xFFFF);
    assert((f2(0xFFFF)  & 0xFFFF ) == 0);
    assert((f2(0xFFFFL)  & 0xFFFF ) == 0);
#endif
}
