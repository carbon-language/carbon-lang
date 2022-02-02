//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: locale.en_US.UTF-8

// <locale>

// template <class charT> class ctype_byname;

// const charT* toupper(charT* low, const charT* high) const;

#include <locale>
#include <string>
#include <cassert>

#include "test_macros.h"
#include "platform_support.h" // locale name macros

int main(int, char**)
{
    {
        std::locale l;
        {
            typedef std::ctype_byname<char> F;
            std::locale ll(l, new F(LOCALE_en_US_UTF_8));
            const F& f = std::use_facet<F>(ll);
            std::string in("c A\x07.a1");

            assert(f.toupper(&in[0], in.data() + in.size()) == in.data() + in.size());
            assert(in[0] == 'C');
            assert(in[1] == ' ');
            assert(in[2] == 'A');
            assert(in[3] == '\x07');
            assert(in[4] == '.');
            assert(in[5] == 'A');
            assert(in[6] == '1');
        }
    }
    {
        std::locale l;
        {
            typedef std::ctype_byname<char> F;
            std::locale ll(l, new F("C"));
            const F& f = std::use_facet<F>(ll);
            std::string in("\xFA A\x07.a1");

            assert(f.toupper(&in[0], in.data() + in.size()) == in.data() + in.size());
            assert(in[0] == '\xFA');
            assert(in[1] == ' ');
            assert(in[2] == 'A');
            assert(in[3] == '\x07');
            assert(in[4] == '.');
            assert(in[5] == 'A');
            assert(in[6] == '1');
        }
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::locale l;
        {
            typedef std::ctype_byname<wchar_t> F;
            std::locale ll(l, new F(LOCALE_en_US_UTF_8));
            const F& f = std::use_facet<F>(ll);
            std::wstring in(L"\xFA A\x07.a1");

            assert(f.toupper(&in[0], in.data() + in.size()) == in.data() + in.size());
            assert(in[0] == L'\xDA');
            assert(in[1] == L' ');
            assert(in[2] == L'A');
            assert(in[3] == L'\x07');
            assert(in[4] == L'.');
            assert(in[5] == L'A');
            assert(in[6] == L'1');
        }
    }
    {
        std::locale l;
        {
            typedef std::ctype_byname<wchar_t> F;
            std::locale ll(l, new F("C"));
            const F& f = std::use_facet<F>(ll);
            std::wstring in(L"\u00FA A\x07.a1");

            assert(f.toupper(&in[0], in.data() + in.size()) == in.data() + in.size());
            assert(in[0] == L'\u00FA');
            assert(in[1] == L' ');
            assert(in[2] == L'A');
            assert(in[3] == L'\x07');
            assert(in[4] == L'.');
            assert(in[5] == L'A');
            assert(in[6] == L'1');
        }
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
