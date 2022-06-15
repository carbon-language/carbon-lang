//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: !stdlib=libc++ && (c++03 || c++11 || c++14)

// <string_view>
// constexpr size_type rfind(charT c, size_type pos = npos) const;

#include <string_view>
#include <cassert>

#include "test_macros.h"
#include "constexpr_char_traits.h"

template <class S>
void
test(const S& s, typename S::value_type c, typename S::size_type pos,
     typename S::size_type x)
{
    LIBCPP_ASSERT_NOEXCEPT(s.rfind(c, pos));
    assert(s.rfind(c, pos) == x);
    if (x != S::npos)
        assert(x <= pos && x + 1 <= s.size());
}

template <class S>
void
test(const S& s, typename S::value_type c, typename S::size_type x)
{
    LIBCPP_ASSERT_NOEXCEPT(s.rfind(c));
    assert(s.rfind(c) == x);
    if (x != S::npos)
        assert(x + 1 <= s.size());
}

int main(int, char**)
{
    {
    typedef std::string_view S;
    test(S(""), 'b', 0, S::npos);
    test(S(""), 'b', 1, S::npos);
    test(S("abcde"), 'b', 0, S::npos);
    test(S("abcde"), 'b', 1, 1);
    test(S("abcde"), 'b', 2, 1);
    test(S("abcde"), 'b', 4, 1);
    test(S("abcde"), 'b', 5, 1);
    test(S("abcde"), 'b', 6, 1);
    test(S("abcdeabcde"), 'b', 0, S::npos);
    test(S("abcdeabcde"), 'b', 1, 1);
    test(S("abcdeabcde"), 'b', 5, 1);
    test(S("abcdeabcde"), 'b', 9, 6);
    test(S("abcdeabcde"), 'b', 10, 6);
    test(S("abcdeabcde"), 'b', 11, 6);
    test(S("abcdeabcdeabcdeabcde"), 'b', 0, S::npos);
    test(S("abcdeabcdeabcdeabcde"), 'b', 1, 1);
    test(S("abcdeabcdeabcdeabcde"), 'b', 10, 6);
    test(S("abcdeabcdeabcdeabcde"), 'b', 19, 16);
    test(S("abcdeabcdeabcdeabcde"), 'b', 20, 16);
    test(S("abcdeabcdeabcdeabcde"), 'b', 21, 16);

    test(S(""), 'b', S::npos);
    test(S("abcde"), 'b', 1);
    test(S("abcdeabcde"), 'b', 6);
    test(S("abcdeabcdeabcdeabcde"), 'b', 16);
    }

#if TEST_STD_VER > 11
    {
    typedef std::basic_string_view<char, constexpr_char_traits<char>> SV;
    constexpr SV  sv1;
    constexpr SV  sv2 { "abcde", 5 };

    static_assert (sv1.rfind( 'b', 0 ) == SV::npos, "" );
    static_assert (sv1.rfind( 'b', 1 ) == SV::npos, "" );
    static_assert (sv2.rfind( 'b', 0 ) == SV::npos, "" );
    static_assert (sv2.rfind( 'b', 1 ) == 1, "" );
    static_assert (sv2.rfind( 'b', 2 ) == 1, "" );
    static_assert (sv2.rfind( 'b', 3 ) == 1, "" );
    static_assert (sv2.rfind( 'b', 4 ) == 1, "" );
    }
#endif

  return 0;
}
