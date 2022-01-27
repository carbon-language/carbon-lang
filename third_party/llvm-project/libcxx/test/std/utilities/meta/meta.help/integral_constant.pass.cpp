//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// integral_constant

#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::integral_constant<int, 5> _5;
    static_assert(_5::value == 5, "");
    static_assert((std::is_same<_5::value_type, int>::value), "");
    static_assert((std::is_same<_5::type, _5>::value), "");
#if TEST_STD_VER >= 11
    static_assert((_5() == 5), "");
#endif
    assert(_5() == 5);


#if TEST_STD_VER > 11
    static_assert ( _5{}() == 5, "" );
    static_assert ( std::true_type{}(), "" );
#endif

    static_assert(std::false_type::value == false, "");
    static_assert((std::is_same<std::false_type::value_type, bool>::value), "");
    static_assert((std::is_same<std::false_type::type, std::false_type>::value), "");

    static_assert(std::true_type::value == true, "");
    static_assert((std::is_same<std::true_type::value_type, bool>::value), "");
    static_assert((std::is_same<std::true_type::type, std::true_type>::value), "");

    std::false_type f1;
    std::false_type f2 = f1;
    assert(!f2);

    std::true_type t1;
    std::true_type t2 = t1;
    assert(t2);

  return 0;
}
