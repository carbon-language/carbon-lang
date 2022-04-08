//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// class match_results<BidirectionalIterator, Allocator>

// template <class OutputIter, class ST, class SA>
//   OutputIter
//   format(OutputIter out, const basic_string<char_type, ST, SA>& fmt,
//          regex_constants::match_flag_type flags = regex_constants::format_default) const;

#include <regex>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "test_allocator.h"

int main(int, char**)
{
    typedef std::basic_string<char, std::char_traits<char>, test_allocator<char> > nstr;
    {
        std::match_results<const char*> m;
        const char s[] = "abcdefghijk";
        assert(std::regex_search(s, m, std::regex("cd((e)fg)hi")));

        char out[100] = {0};
        nstr fmt("prefix: $`, match: $&, suffix: $', m[1]: $1, m[2]: $2");
        char* r = m.format(cpp17_output_iterator<char*>(out), fmt).base();
        assert(r == out + 58);
        assert(std::string(out) == "prefix: ab, match: cdefghi, suffix: jk, m[1]: efg, m[2]: e");
    }
    {
        std::match_results<const char*> m;
        const char s[] = "abcdefghijk";
        assert(std::regex_search(s, m, std::regex("cd((e)fg)hi")));

        char out[100] = {0};
        nstr fmt("prefix: $`, match: $&, suffix: $', m[1]: $1, m[2]: $2");
        char* r = m.format(cpp17_output_iterator<char*>(out),
                    fmt, std::regex_constants::format_sed).base();
        assert(r == out + 59);
        assert(std::string(out) == "prefix: $`, match: $cdefghi, suffix: $', m[1]: $1, m[2]: $2");
    }
    {
        std::match_results<const char*> m;
        const char s[] = "abcdefghijk";
        assert(std::regex_search(s, m, std::regex("cd((e)fg)hi")));

        char out[100] = {0};
        nstr fmt("match: &, m[1]: \\1, m[2]: \\2");
        char* r = m.format(cpp17_output_iterator<char*>(out),
                    fmt, std::regex_constants::format_sed).base();
        assert(r == out + 34);
        assert(std::string(out) == "match: cdefghi, m[1]: efg, m[2]: e");
    }

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    typedef std::basic_string<wchar_t, std::char_traits<wchar_t>, test_allocator<wchar_t> > wstr;
    {
        std::match_results<const wchar_t*> m;
        const wchar_t s[] = L"abcdefghijk";
        assert(std::regex_search(s, m, std::wregex(L"cd((e)fg)hi")));

        wchar_t out[100] = {0};
        wstr fmt(L"prefix: $`, match: $&, suffix: $', m[1]: $1, m[2]: $2");
        wchar_t* r = m.format(cpp17_output_iterator<wchar_t*>(out), fmt).base();
        assert(r == out + 58);
        assert(std::wstring(out) == L"prefix: ab, match: cdefghi, suffix: jk, m[1]: efg, m[2]: e");
    }
    {
        std::match_results<const wchar_t*> m;
        const wchar_t s[] = L"abcdefghijk";
        assert(std::regex_search(s, m, std::wregex(L"cd((e)fg)hi")));

        wchar_t out[100] = {0};
        wstr fmt(L"prefix: $`, match: $&, suffix: $', m[1]: $1, m[2]: $2");
        wchar_t* r = m.format(cpp17_output_iterator<wchar_t*>(out),
                    fmt, std::regex_constants::format_sed).base();
        assert(r == out + 59);
        assert(std::wstring(out) == L"prefix: $`, match: $cdefghi, suffix: $', m[1]: $1, m[2]: $2");
    }
    {
        std::match_results<const wchar_t*> m;
        const wchar_t s[] = L"abcdefghijk";
        assert(std::regex_search(s, m, std::wregex(L"cd((e)fg)hi")));

        wchar_t out[100] = {0};
        wstr fmt(L"match: &, m[1]: \\1, m[2]: \\2");
        wchar_t* r = m.format(cpp17_output_iterator<wchar_t*>(out),
                    fmt, std::regex_constants::format_sed).base();
        assert(r == out + 34);
        assert(std::wstring(out) == L"match: cdefghi, m[1]: efg, m[2]: e");
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
