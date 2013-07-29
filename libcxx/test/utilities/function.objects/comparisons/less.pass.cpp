//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// less

#include <functional>
#include <type_traits>
#include <cassert>

int main()
{
    typedef std::less<int> F;
    const F f = F();
    static_assert((std::is_base_of<std::binary_function<int, int, bool>, F>::value), "");
    assert(!f(36, 36));
    assert(!f(36, 6));
    assert(f(6, 36));
#if _LIBCPP_STD_VER > 11
    typedef std::less<> F2;
    const F2 f2 = F2();
    assert(!f2(36, 36));
    assert(!f2(36, 6));
    assert( f2(6, 36));
    assert(!f2(36, 6.0));
    assert(!f2(36.0, 6));
    assert( f2(6, 36.0));
    assert( f2(6.0, 36));
#endif
}
