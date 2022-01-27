//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: locale.en_US.UTF-8

// <iomanip>

// template <class charT> T9 get_time(struct tm* tmb, const charT* fmt);

#include <iomanip>
#include <istream>
#include <cassert>

#include "test_macros.h"
#include "platform_support.h" // locale name macros

template <class CharT>
struct testbuf
    : public std::basic_streambuf<CharT>
{
    typedef std::basic_string<CharT> string_type;
    typedef std::basic_streambuf<CharT> base;
private:
    string_type str_;
public:

    testbuf() {}
    testbuf(const string_type& str)
        : str_(str)
    {
        base::setg(const_cast<CharT*>(str_.data()),
                   const_cast<CharT*>(str_.data()),
                   const_cast<CharT*>(str_.data()) + str_.size());
    }
};

int main(int, char**)
{
    {
        testbuf<char> sb("  Sat Dec 31 23:55:59 2061");
        std::istream is(&sb);
        is.imbue(std::locale(LOCALE_en_US_UTF_8));
        std::tm t = {};
        is >> std::get_time(&t, "%a %b %d %H:%M:%S %Y");
        assert(t.tm_sec == 59);
        assert(t.tm_min == 55);
        assert(t.tm_hour == 23);
        assert(t.tm_mday == 31);
        assert(t.tm_mon == 11);
        assert(t.tm_year == 161);
        assert(t.tm_wday == 6);
        assert(is.eof());
        assert(!is.fail());
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        testbuf<wchar_t> sb(L"  Sat Dec 31 23:55:59 2061");
        std::wistream is(&sb);
        is.imbue(std::locale(LOCALE_en_US_UTF_8));
        std::tm t = {};
        is >> std::get_time(&t, L"%a %b %d %H:%M:%S %Y");
        assert(t.tm_sec == 59);
        assert(t.tm_min == 55);
        assert(t.tm_hour == 23);
        assert(t.tm_mday == 31);
        assert(t.tm_mon == 11);
        assert(t.tm_year == 161);
        assert(t.tm_wday == 6);
        assert(is.eof());
        assert(!is.fail());
    }
#endif

  return 0;
}
