//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT> >
// class basic_stringbuf

// int_type underflow();

#include <sstream>
#include <cassert>

#include "test_macros.h"

template <class CharT>
struct testbuf
    : public std::basic_stringbuf<CharT>
{
    typedef std::basic_stringbuf<CharT> base;
    explicit testbuf(const std::basic_string<CharT>& str)
        : base(str) {}

    typename base::int_type underflow() {return base::underflow();}
    void pbump(int n) {base::pbump(n);}
};

int main(int, char**)
{
    {
        testbuf<char> sb("123");
        sb.pbump(3);
        assert(sb.underflow() == '1');
        assert(sb.underflow() == '1');
        assert(sb.snextc() == '2');
        assert(sb.underflow() == '2');
        assert(sb.underflow() == '2');
        assert(sb.snextc() == '3');
        assert(sb.underflow() == '3');
        assert(sb.underflow() == '3');
        assert(sb.snextc() == std::char_traits<char>::eof());
        assert(sb.underflow() == std::char_traits<char>::eof());
        assert(sb.underflow() == std::char_traits<char>::eof());
        sb.sputc('4');
        assert(sb.underflow() == '4');
        assert(sb.underflow() == '4');
    }
    {
        testbuf<wchar_t> sb(L"123");
        sb.pbump(3);
        assert(sb.underflow() == L'1');
        assert(sb.underflow() == L'1');
        assert(sb.snextc() == L'2');
        assert(sb.underflow() == L'2');
        assert(sb.underflow() == L'2');
        assert(sb.snextc() == L'3');
        assert(sb.underflow() == L'3');
        assert(sb.underflow() == L'3');
        assert(sb.snextc() == std::char_traits<wchar_t>::eof());
        assert(sb.underflow() == std::char_traits<wchar_t>::eof());
        assert(sb.underflow() == std::char_traits<wchar_t>::eof());
        sb.sputc(L'4');
        assert(sb.underflow() == L'4');
        assert(sb.underflow() == L'4');
    }

  return 0;
}
