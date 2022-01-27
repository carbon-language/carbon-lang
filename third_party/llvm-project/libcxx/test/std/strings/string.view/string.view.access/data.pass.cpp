//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string_view>

// constexpr const _CharT* data() const noexcept;

#include <string_view>
#include <cassert>

#include "test_macros.h"

template <typename CharT>
void test ( const CharT *s, size_t len ) {
    std::basic_string_view<CharT> sv ( s, len );
    assert ( sv.length() == len );
    assert ( sv.data() == s );
#if TEST_STD_VER > 14
//  make sure we pick up std::data, too!
    assert ( sv.data() == std::data(sv));
#endif
    }

int main(int, char**) {
    test ( "ABCDE", 5 );
    test ( "a", 1 );

    test ( L"ABCDE", 5 );
    test ( L"a", 1 );

#if TEST_STD_VER >= 11
    test ( u"ABCDE", 5 );
    test ( u"a", 1 );

    test ( U"ABCDE", 5 );
    test ( U"a", 1 );
#endif

#if TEST_STD_VER > 11
    {
    constexpr const char *s = "ABC";
    constexpr std::basic_string_view<char> sv( s, 2 );
    static_assert( sv.length() ==  2,  "" );
    static_assert( sv.data() == s, "" );
    }
#endif

  return 0;
}
