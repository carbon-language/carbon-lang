//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string_view>

// constexpr template<class charT, class traits, class Allocator>
//   bool operator>(const charT* lhs, basic_string_view<charT,traits> rhs);
// constexpr template<class charT, class traits, class Allocator>
//   bool operator>(basic_string_view<charT,traits> lhs, const charT* rhs);

#include <string_view>
#include <cassert>

#include "test_macros.h"
#include "constexpr_char_traits.h"

template <class S>
void
test(const typename S::value_type* lhs, const S& rhs, bool x, bool y)
{
    assert((lhs > rhs) == x);
    assert((rhs > lhs) == y);
}

int main(int, char**)
{
    {
    typedef std::string_view S;
    test("", S(""), false, false);
    test("", S("abcde"), false, true);
    test("", S("abcdefghij"), false, true);
    test("", S("abcdefghijklmnopqrst"), false, true);
    test("abcde", S(""), true, false);
    test("abcde", S("abcde"), false, false);
    test("abcde", S("abcdefghij"), false, true);
    test("abcde", S("abcdefghijklmnopqrst"), false, true);
    test("abcdefghij", S(""), true, false);
    test("abcdefghij", S("abcde"), true, false);
    test("abcdefghij", S("abcdefghij"), false, false);
    test("abcdefghij", S("abcdefghijklmnopqrst"), false, true);
    test("abcdefghijklmnopqrst", S(""), true, false);
    test("abcdefghijklmnopqrst", S("abcde"), true, false);
    test("abcdefghijklmnopqrst", S("abcdefghij"), true, false);
    test("abcdefghijklmnopqrst", S("abcdefghijklmnopqrst"), false, false);
    }

#if TEST_STD_VER > 11
    {
    typedef std::basic_string_view<char, constexpr_char_traits<char>> SV;
    constexpr SV  sv1;
    constexpr SV  sv2 { "abcde", 5 };

    static_assert (!(sv1     > ""), "" );
    static_assert (!(""      > sv1), "" );
    static_assert (!(sv1     > "abcde"), "" );
    static_assert (  "abcde" > sv1, "" );

    static_assert (  sv2      > "", "" );
    static_assert (!(""       > sv2), "" );
    static_assert (!(sv2      > "abcde"), "" );
    static_assert (!("abcde"  > sv2), "" );
    static_assert (!(sv2      > "abcde0"), "" );
    static_assert (  "abcde0" > sv2, "" );
    }
#endif

  return 0;
}
