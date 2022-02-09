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

// const charT* tolower(charT* low, const charT* high) const;

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

            assert(f.tolower(&in[0], in.data() + in.size()) == in.data() + in.size());
            assert(in[0] == 'c');
            assert(in[1] == ' ');
            assert(in[2] == 'a');
            assert(in[3] == '\x07');
            assert(in[4] == '.');
            assert(in[5] == 'a');
            assert(in[6] == '1');
        }
    }
    {
        std::locale l;
        {
            typedef std::ctype_byname<char> F;
            std::locale ll(l, new F("C"));
            const F& f = std::use_facet<F>(ll);
            std::string in("\xDA A\x07.a1");

            assert(f.tolower(&in[0], in.data() + in.size()) == in.data() + in.size());
            assert(in[0] == '\xDA');
            assert(in[1] == ' ');
            assert(in[2] == 'a');
            assert(in[3] == '\x07');
            assert(in[4] == '.');
            assert(in[5] == 'a');
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
            std::wstring in(L"\xDA A\x07.a1");

            assert(f.tolower(&in[0], in.data() + in.size()) == in.data() + in.size());
            assert(in[0] == L'\xFA');
            assert(in[1] == L' ');
            assert(in[2] == L'a');
            assert(in[3] == L'\x07');
            assert(in[4] == L'.');
            assert(in[5] == L'a');
            assert(in[6] == L'1');
        }
    }
    {
        std::locale l;
        {
            typedef std::ctype_byname<wchar_t> F;
            std::locale ll(l, new F("C"));
            const F& f = std::use_facet<F>(ll);
            std::wstring in(L"\xDA A\x07.a1");

            assert(f.tolower(&in[0], in.data() + in.size()) == in.data() + in.size());
            assert(in[0] == L'\xDA');
            assert(in[1] == L' ');
            assert(in[2] == L'a');
            assert(in[3] == L'\x07');
            assert(in[4] == L'.');
            assert(in[5] == L'a');
            assert(in[6] == L'1');
        }
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
