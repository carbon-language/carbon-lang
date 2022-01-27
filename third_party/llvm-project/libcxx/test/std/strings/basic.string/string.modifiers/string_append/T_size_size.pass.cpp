//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template <class T>
//    basic_string& append(const T& t, size_type pos, size_type n=npos); // C++17

#include <string>
#include <string>
#include <stdexcept>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S, class SV>
void
test(S s, SV sv, typename S::size_type pos, typename S::size_type n, S expected)
{
    if (pos <= sv.size())
    {
        s.append(sv, pos, n);
        LIBCPP_ASSERT(s.__invariants());
        assert(s == expected);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    else
    {
        try
        {
            s.append(sv, pos, n);
            assert(false);
        }
        catch (std::out_of_range&)
        {
            assert(pos > sv.size());
        }
    }
#endif
}

template <class S, class SV>
void
test_npos(S s, SV sv, typename S::size_type pos, S expected)
{
    if (pos <= sv.size())
    {
        s.append(sv, pos);
        LIBCPP_ASSERT(s.__invariants());
        assert(s == expected);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    else
    {
        try
        {
            s.append(sv, pos);
            assert(false);
        }
        catch (std::out_of_range&)
        {
            assert(pos > sv.size());
        }
    }
#endif
}

int main(int, char**)
{
    {
    typedef std::string S;
    typedef std::string_view SV;
    test(S(), SV(), 0, 0, S());
    test(S(), SV(), 1, 0, S());
    test(S(), SV("12345"), 0, 3, S("123"));
    test(S(), SV("12345"), 1, 4, S("2345"));
    test(S(), SV("12345"), 3, 15, S("45"));
    test(S(), SV("12345"), 5, 15, S(""));
    test(S(), SV("12345"), 6, 15, S("not happening"));
    test(S(), SV("12345678901234567890"), 0, 0, S());
    test(S(), SV("12345678901234567890"), 1, 1, S("2"));
    test(S(), SV("12345678901234567890"), 2, 3, S("345"));
    test(S(), SV("12345678901234567890"), 12, 13, S("34567890"));
    test(S(), SV("12345678901234567890"), 21, 13, S("not happening"));

    test(S("12345"), SV(), 0, 0, S("12345"));
    test(S("12345"), SV("12345"), 2, 2, S("1234534"));
    test(S("12345"), SV("1234567890"), 0, 100, S("123451234567890"));

    test(S("12345678901234567890"), SV(), 0, 0, S("12345678901234567890"));
    test(S("12345678901234567890"), SV("12345"), 1, 3, S("12345678901234567890234"));
    test(S("12345678901234567890"), SV("12345678901234567890"), 5, 10,
         S("123456789012345678906789012345"));
    }
#if TEST_STD_VER >= 11
    {
    typedef std::basic_string     <char, std::char_traits<char>, min_allocator<char>> S;
    typedef std::basic_string_view<char, std::char_traits<char> > SV;
    test(S(), SV(), 0, 0, S());
    test(S(), SV(), 1, 0, S());
    test(S(), SV("12345"), 0, 3, S("123"));
    test(S(), SV("12345"), 1, 4, S("2345"));
    test(S(), SV("12345"), 3, 15, S("45"));
    test(S(), SV("12345"), 5, 15, S(""));
    test(S(), SV("12345"), 6, 15, S("not happening"));
    test(S(), SV("12345678901234567890"), 0, 0, S());
    test(S(), SV("12345678901234567890"), 1, 1, S("2"));
    test(S(), SV("12345678901234567890"), 2, 3, S("345"));
    test(S(), SV("12345678901234567890"), 12, 13, S("34567890"));
    test(S(), SV("12345678901234567890"), 21, 13, S("not happening"));

    test(S("12345"), SV(), 0, 0, S("12345"));
    test(S("12345"), SV("12345"), 2, 2, S("1234534"));
    test(S("12345"), SV("1234567890"), 0, 100, S("123451234567890"));

    test(S("12345678901234567890"), SV(), 0, 0, S("12345678901234567890"));
    test(S("12345678901234567890"), SV("12345"), 1, 3, S("12345678901234567890234"));
    test(S("12345678901234567890"), SV("12345678901234567890"), 5, 10,
         S("123456789012345678906789012345"));
    }
#endif
    {
    typedef std::string S;
    typedef std::string_view SV;
    test_npos(S(), SV(), 0, S());
    test_npos(S(), SV(), 1, S());
    test_npos(S(), SV("12345"), 0, S("12345"));
    test_npos(S(), SV("12345"), 1, S("2345"));
    test_npos(S(), SV("12345"), 3, S("45"));
    test_npos(S(), SV("12345"), 5, S(""));
    test_npos(S(), SV("12345"), 6, S("not happening"));
    }

    {
    std::string s;
    std::string_view sv = "EFGH";
    char arr[] = "IJKL";

    s.append("CDEF", 0);    // calls append(const char *, len)
    assert(s == "");
    s.clear();

    s.append("QRST", 0, std::string::npos); // calls append(string("QRST"), pos, npos)
    assert(s == "QRST");
    s.clear();

    s.append(sv, 0);  // calls append(T, pos, npos)
    assert(s == sv);
    s.clear();

    s.append(sv, 0, std::string::npos);   // calls append(T, pos, npos)
    assert(s == sv);
    s.clear();

    s.append(arr, 0);     // calls append(const char *, len)
    assert(s == "");
    s.clear();

    s.append(arr, 0, std::string::npos);    // calls append(string("IJKL"), pos, npos)
    assert(s == "IJKL");
    s.clear();

    s.append(arr, 0);     // calls append(const char *, len)
    assert(s == "");
    s.clear();
    }

    {
    std::string s = "ABCD";
    std::string_view sv = s;
    s.append(sv);
    assert(s == "ABCDABCD");

    sv = s;
    s.append(sv, 0, std::string::npos);
    assert(s == "ABCDABCDABCDABCD");

    sv = s;
    s.append(sv, sv.size());
    assert(s == "ABCDABCDABCDABCD");
    }

    {
    std::string s = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::string_view sv = s;
    s.append(sv);
    assert(s == "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ");

    sv = s;
    s.append(sv, 0, std::string::npos);
    assert(s == "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ");
    }

  return 0;
}
