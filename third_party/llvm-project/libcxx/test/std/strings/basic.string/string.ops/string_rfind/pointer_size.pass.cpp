//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// size_type rfind(const charT* s, size_type pos = npos) const;

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
void
test(const S& s, const typename S::value_type* str, typename S::size_type pos,
     typename S::size_type x)
{
    LIBCPP_ASSERT_NOEXCEPT(s.rfind(str, pos));
    assert(s.rfind(str, pos) == x);
    if (x != S::npos)
    {
        typename S::size_type n = S::traits_type::length(str);
        assert(x <= pos && x + n <= s.size());
    }
}

template <class S>
void
test(const S& s, const typename S::value_type* str, typename S::size_type x)
{
    LIBCPP_ASSERT_NOEXCEPT(s.rfind(str));
    assert(s.rfind(str) == x);
    if (x != S::npos)
    {
        typename S::size_type pos = s.size();
        typename S::size_type n = S::traits_type::length(str);
        assert(x <= pos && x + n <= s.size());
    }
}

template <class S>
void test0()
{
    test(S(""), "", 0, 0);
    test(S(""), "abcde", 0, S::npos);
    test(S(""), "abcdeabcde", 0, S::npos);
    test(S(""), "abcdeabcdeabcdeabcde", 0, S::npos);
    test(S(""), "", 1, 0);
    test(S(""), "abcde", 1, S::npos);
    test(S(""), "abcdeabcde", 1, S::npos);
    test(S(""), "abcdeabcdeabcdeabcde", 1, S::npos);
    test(S("abcde"), "", 0, 0);
    test(S("abcde"), "abcde", 0, 0);
    test(S("abcde"), "abcdeabcde", 0, S::npos);
    test(S("abcde"), "abcdeabcdeabcdeabcde", 0, S::npos);
    test(S("abcde"), "", 1, 1);
    test(S("abcde"), "abcde", 1, 0);
    test(S("abcde"), "abcdeabcde", 1, S::npos);
    test(S("abcde"), "abcdeabcdeabcdeabcde", 1, S::npos);
    test(S("abcde"), "", 2, 2);
    test(S("abcde"), "abcde", 2, 0);
    test(S("abcde"), "abcdeabcde", 2, S::npos);
    test(S("abcde"), "abcdeabcdeabcdeabcde", 2, S::npos);
    test(S("abcde"), "", 4, 4);
    test(S("abcde"), "abcde", 4, 0);
    test(S("abcde"), "abcdeabcde", 4, S::npos);
    test(S("abcde"), "abcdeabcdeabcdeabcde", 4, S::npos);
    test(S("abcde"), "", 5, 5);
    test(S("abcde"), "abcde", 5, 0);
    test(S("abcde"), "abcdeabcde", 5, S::npos);
    test(S("abcde"), "abcdeabcdeabcdeabcde", 5, S::npos);
    test(S("abcde"), "", 6, 5);
    test(S("abcde"), "abcde", 6, 0);
    test(S("abcde"), "abcdeabcde", 6, S::npos);
    test(S("abcde"), "abcdeabcdeabcdeabcde", 6, S::npos);
    test(S("abcdeabcde"), "", 0, 0);
    test(S("abcdeabcde"), "abcde", 0, 0);
    test(S("abcdeabcde"), "abcdeabcde", 0, 0);
    test(S("abcdeabcde"), "abcdeabcdeabcdeabcde", 0, S::npos);
    test(S("abcdeabcde"), "", 1, 1);
    test(S("abcdeabcde"), "abcde", 1, 0);
    test(S("abcdeabcde"), "abcdeabcde", 1, 0);
    test(S("abcdeabcde"), "abcdeabcdeabcdeabcde", 1, S::npos);
    test(S("abcdeabcde"), "", 5, 5);
    test(S("abcdeabcde"), "abcde", 5, 5);
    test(S("abcdeabcde"), "abcdeabcde", 5, 0);
    test(S("abcdeabcde"), "abcdeabcdeabcdeabcde", 5, S::npos);
    test(S("abcdeabcde"), "", 9, 9);
    test(S("abcdeabcde"), "abcde", 9, 5);
    test(S("abcdeabcde"), "abcdeabcde", 9, 0);
    test(S("abcdeabcde"), "abcdeabcdeabcdeabcde", 9, S::npos);
    test(S("abcdeabcde"), "", 10, 10);
    test(S("abcdeabcde"), "abcde", 10, 5);
    test(S("abcdeabcde"), "abcdeabcde", 10, 0);
    test(S("abcdeabcde"), "abcdeabcdeabcdeabcde", 10, S::npos);
    test(S("abcdeabcde"), "", 11, 10);
    test(S("abcdeabcde"), "abcde", 11, 5);
    test(S("abcdeabcde"), "abcdeabcde", 11, 0);
    test(S("abcdeabcde"), "abcdeabcdeabcdeabcde", 11, S::npos);
    test(S("abcdeabcdeabcdeabcde"), "", 0, 0);
    test(S("abcdeabcdeabcdeabcde"), "abcde", 0, 0);
    test(S("abcdeabcdeabcdeabcde"), "abcdeabcde", 0, 0);
    test(S("abcdeabcdeabcdeabcde"), "abcdeabcdeabcdeabcde", 0, 0);
    test(S("abcdeabcdeabcdeabcde"), "", 1, 1);
    test(S("abcdeabcdeabcdeabcde"), "abcde", 1, 0);
    test(S("abcdeabcdeabcdeabcde"), "abcdeabcde", 1, 0);
    test(S("abcdeabcdeabcdeabcde"), "abcdeabcdeabcdeabcde", 1, 0);
    test(S("abcdeabcdeabcdeabcde"), "", 10, 10);
    test(S("abcdeabcdeabcdeabcde"), "abcde", 10, 10);
    test(S("abcdeabcdeabcdeabcde"), "abcdeabcde", 10, 10);
    test(S("abcdeabcdeabcdeabcde"), "abcdeabcdeabcdeabcde", 10, 0);
    test(S("abcdeabcdeabcdeabcde"), "", 19, 19);
    test(S("abcdeabcdeabcdeabcde"), "abcde", 19, 15);
    test(S("abcdeabcdeabcdeabcde"), "abcdeabcde", 19, 10);
    test(S("abcdeabcdeabcdeabcde"), "abcdeabcdeabcdeabcde", 19, 0);
    test(S("abcdeabcdeabcdeabcde"), "", 20, 20);
    test(S("abcdeabcdeabcdeabcde"), "abcde", 20, 15);
    test(S("abcdeabcdeabcdeabcde"), "abcdeabcde", 20, 10);
    test(S("abcdeabcdeabcdeabcde"), "abcdeabcdeabcdeabcde", 20, 0);
    test(S("abcdeabcdeabcdeabcde"), "", 21, 20);
    test(S("abcdeabcdeabcdeabcde"), "abcde", 21, 15);
    test(S("abcdeabcdeabcdeabcde"), "abcdeabcde", 21, 10);
    test(S("abcdeabcdeabcdeabcde"), "abcdeabcdeabcdeabcde", 21, 0);
}

template <class S>
void test1()
{
    test(S(""), "", 0);
    test(S(""), "abcde", S::npos);
    test(S(""), "abcdeabcde", S::npos);
    test(S(""), "abcdeabcdeabcdeabcde", S::npos);
    test(S("abcde"), "", 5);
    test(S("abcde"), "abcde", 0);
    test(S("abcde"), "abcdeabcde", S::npos);
    test(S("abcde"), "abcdeabcdeabcdeabcde", S::npos);
    test(S("abcdeabcde"), "", 10);
    test(S("abcdeabcde"), "abcde", 5);
    test(S("abcdeabcde"), "abcdeabcde", 0);
    test(S("abcdeabcde"), "abcdeabcdeabcdeabcde", S::npos);
    test(S("abcdeabcdeabcdeabcde"), "", 20);
    test(S("abcdeabcdeabcdeabcde"), "abcde", 15);
    test(S("abcdeabcdeabcdeabcde"), "abcdeabcde", 10);
    test(S("abcdeabcdeabcdeabcde"), "abcdeabcdeabcdeabcde", 0);
}

int main(int, char**)
{
    {
    typedef std::string S;
    test0<S>();
    test1<S>();
    }
#if TEST_STD_VER >= 11
    {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    test0<S>();
    test1<S>();
    }
#endif

  return 0;
}
