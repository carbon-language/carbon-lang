//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <functional>

// bit_and

#include <functional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::bit_and<int> F;
    const F f = F();
#if TEST_STD_VER <= 17
    static_assert((std::is_same<int, F::first_argument_type>::value), "" );
    static_assert((std::is_same<int, F::second_argument_type>::value), "" );
    static_assert((std::is_same<int, F::result_type>::value), "" );
#endif
    assert(f(0xEA95, 0xEA95) == 0xEA95);
    assert(f(0xEA95, 0x58D3) == 0x4891);
    assert(f(0x58D3, 0xEA95) == 0x4891);
    assert(f(0x58D3, 0) == 0);
    assert(f(0xFFFF, 0x58D3) == 0x58D3);
#if TEST_STD_VER > 11
    typedef std::bit_and<> F2;
    const F2 f2 = F2();
    assert(f2(0xEA95, 0xEA95) == 0xEA95);
    assert(f2(0xEA95L, 0xEA95) == 0xEA95);
    assert(f2(0xEA95, 0xEA95L) == 0xEA95);

    assert(f2(0xEA95, 0x58D3) == 0x4891);
    assert(f2(0xEA95L, 0x58D3) == 0x4891);
    assert(f2(0xEA95, 0x58D3L) == 0x4891);

    assert(f2(0x58D3, 0xEA95) == 0x4891);
    assert(f2(0x58D3L, 0xEA95) == 0x4891);
    assert(f2(0x58D3, 0xEA95L) == 0x4891);

    assert(f2(0x58D3, 0) == 0);
    assert(f2(0x58D3L, 0) == 0);
    assert(f2(0x58D3, 0L) == 0);

    assert(f2(0xFFFF, 0x58D3) == 0x58D3);
    assert(f2(0xFFFFL, 0x58D3) == 0x58D3);
    assert(f2(0xFFFF, 0x58D3L) == 0x58D3);

    constexpr int foo = std::bit_and<int> () (0x58D3, 0xEA95);
    static_assert ( foo == 0x4891, "" );

    constexpr int bar = std::bit_and<> () (0x58D3L, 0xEA95);
    static_assert ( bar == 0x4891, "" );
#endif

  return 0;
}
