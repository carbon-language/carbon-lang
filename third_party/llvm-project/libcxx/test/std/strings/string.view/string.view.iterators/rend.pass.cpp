//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: !stdlib=libc++ && (c++03 || c++11 || c++14)

// <string_view>

// constexpr const_iterator rend() const;

#include <string_view>
#include <cassert>
#include <cstddef>

#include "test_macros.h"

template <class S>
void
test(S s)
{
    const S& cs = s;
    typename S::reverse_iterator e = s.rend();
    typename S::const_reverse_iterator ce1 = cs.rend();
    typename S::const_reverse_iterator ce2 = s.crend();

    if (s.empty())
    {
        assert(  e ==  s.rbegin());
        assert(ce1 == cs.rbegin());
        assert(ce2 ==  s.rbegin());
    }
    else
    {
        assert(  e !=  s.rbegin());
        assert(ce1 != cs.rbegin());
        assert(ce2 !=  s.rbegin());
    }

    assert(static_cast<std::size_t>(  e -  s.rbegin()) == s.size());
    assert(static_cast<std::size_t>(ce1 - cs.rbegin()) == cs.size());
    assert(static_cast<std::size_t>(ce2 - s.crbegin()) == s.size());

    assert(  e == ce1);
    assert(  e == ce2);
    assert(ce1 == ce2);
}


int main(int, char**)
{
    typedef std::string_view    string_view;
#if defined(__cpp_lib_char8_t) && __cpp_lib_char8_t >= 201811L
    typedef std::u8string_view u8string_view;
#endif
    typedef std::u16string_view u16string_view;
    typedef std::u32string_view u32string_view;

    test(string_view   ());
    test(u16string_view());
    test(u32string_view());
    test(string_view   ( "123"));
#if defined(__cpp_lib_char8_t) && __cpp_lib_char8_t >= 201811L
    test(u8string_view{u8"123"});
#endif
#if TEST_STD_VER >= 11
    test(u16string_view{u"123"});
    test(u32string_view{U"123"});
#endif

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    typedef std::wstring_view   wstring_view;
    test(wstring_view  ());
    test(wstring_view  (L"123"));
#endif

#if TEST_STD_VER > 14
    {
    constexpr string_view       sv { "123", 3 };
#if defined(__cpp_lib_char8_t) && __cpp_lib_char8_t >= 201811L
    constexpr u8string_view u8sv {u8"123", 3 };
#endif
    constexpr u16string_view u16sv {u"123", 3 };
    constexpr u32string_view u32sv {U"123", 3 };

    static_assert (    *--sv.rend() ==    sv[0], "" );
#if defined(__cpp_lib_char8_t) && __cpp_lib_char8_t >= 201811L
    static_assert (  *--u8sv.rend() ==  u8sv[0], "" );
#endif
    static_assert ( *--u16sv.rend() == u16sv[0], "" );
    static_assert ( *--u32sv.rend() == u32sv[0], "" );

    static_assert (    *--sv.crend() ==    sv[0], "" );
#if defined(__cpp_lib_char8_t) && __cpp_lib_char8_t >= 201811L
    static_assert (  *--u8sv.crend() ==  u8sv[0], "" );
#endif
    static_assert ( *--u16sv.crend() == u16sv[0], "" );
    static_assert ( *--u32sv.crend() == u32sv[0], "" );

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        {
            constexpr wstring_view     wsv {L"123", 3 };
            static_assert (   *--wsv.rend() ==   wsv[0], "" );
            static_assert (   *--wsv.crend() ==   wsv[0], "" );
        }
#endif
    }
#endif // TEST_STD_VER > 14

  return 0;
}
