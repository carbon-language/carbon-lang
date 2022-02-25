//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// In macosx10.9 to macosx10.14, streams are provided in the dylib AND they
// have a bug in how they handle null-termination in case of errors (see D40677).
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14}}

// <istream>

// basic_istream<charT,traits>& get(char_type* s, streamsize n);

#include <istream>
#include <cassert>

#include "test_macros.h"

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

    CharT* eback() const {return base::eback();}
    CharT* gptr() const {return base::gptr();}
    CharT* egptr() const {return base::egptr();}
};

int main(int, char**)
{
    {
        testbuf<char> sb("  \n    \n ");
        std::istream is(&sb);
        char s[5];
        is.get(s, 5);
        assert(!is.eof());
        assert(!is.fail());
        assert(std::string(s) == "  ");
        assert(is.gcount() == 2);
        is.get(s, 5);
        assert(!is.eof());
        assert( is.fail());
        assert(std::string(s) == "");
        assert(is.gcount() == 0);
        is.clear();
        assert(is.get() == '\n');
        is.get(s, 5);
        assert(!is.eof());
        assert(!is.fail());
        assert(std::string(s) == "    ");
        assert(is.gcount() == 4);
        assert(is.get() == '\n');
        is.get(s, 5);
        assert( is.eof());
        assert(!is.fail());
        assert(std::string(s) == " ");
        assert(is.gcount() == 1);
        // Check that even in error case the buffer is properly 0-terminated.
        is.get(s, 5);
        assert( is.eof());
        assert( is.fail());
        assert(std::string(s) == "");
        assert(is.gcount() == 0);
    }
    {
        testbuf<wchar_t> sb(L"  \n    \n ");
        std::wistream is(&sb);
        wchar_t s[5];
        is.get(s, 5);
        assert(!is.eof());
        assert(!is.fail());
        assert(std::wstring(s) == L"  ");
        assert(is.gcount() == 2);
        is.get(s, 5);
        assert(!is.eof());
        assert( is.fail());
        assert(std::wstring(s) == L"");
        assert(is.gcount() == 0);
        is.clear();
        assert(is.get() == L'\n');
        is.get(s, 5);
        assert(!is.eof());
        assert(!is.fail());
        assert(std::wstring(s) == L"    ");
        assert(is.gcount() == 4);
        assert(is.get() == L'\n');
        is.get(s, 5);
        assert( is.eof());
        assert(!is.fail());
        assert(std::wstring(s) == L" ");
        assert(is.gcount() == 1);
        // Check that even in error case the buffer is properly 0-terminated.
        is.get(s, 5);
        assert( is.eof());
        assert( is.fail());
        assert(std::wstring(s) == L"");
        assert(is.gcount() == 0);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        testbuf<char> sb(" ");
        std::basic_istream<char> is(&sb);
        char s[5] = "test";
        is.exceptions(std::ios_base::eofbit);
        bool threw = false;
        try {
            is.get(s, 5);
        } catch (std::ios_base::failure&) {
            threw = true;
        }
        assert(!is.bad());
        assert( is.eof());
        assert(!is.fail());
        assert(threw);
        assert(std::string(s) == " ");
        assert(is.gcount() == 1);
    }
    {
        testbuf<wchar_t> sb(L" ");
        std::basic_istream<wchar_t> is(&sb);
        wchar_t s[5] = L"test";
        is.exceptions(std::ios_base::eofbit);
        bool threw = false;
        try {
            is.get(s, 5);
        } catch (std::ios_base::failure&) {
            threw = true;
        }
        assert(!is.bad());
        assert( is.eof());
        assert(!is.fail());
        assert(threw);
        assert(std::wstring(s) == L" ");
        assert(is.gcount() == 1);
    }

    {
        testbuf<char> sb;
        std::basic_istream<char> is(&sb);
        char s[5] = "test";
        is.exceptions(std::ios_base::failbit);
        bool threw = false;
        try {
            is.get(s, 5);
        } catch (std::ios_base::failure&) {
            threw = true;
        }
        assert(!is.bad());
        assert( is.eof());
        assert( is.fail());
        assert(threw);
        assert(std::basic_string<char>(s) == "");
        assert(is.gcount() == 0);
    }
    {
        testbuf<wchar_t> sb;
        std::basic_istream<wchar_t> is(&sb);
        wchar_t s[5] = L"test";
        is.exceptions(std::ios_base::failbit);
        bool threw = false;
        try {
            is.get(s, 5);
        } catch (std::ios_base::failure&) {
            threw = true;
        }
        assert(!is.bad());
        assert( is.eof());
        assert( is.fail());
        assert(threw);
        assert(std::basic_string<wchar_t>(s) == L"");
        assert(is.gcount() == 0);
    }
#endif

  return 0;
}
