//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <istream>

// template<class traits>
//   basic_istream<char,traits>& operator>>(basic_istream<char,traits>&& in, unsigned char& c);

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
        testbuf<char> sb("          ");
        std::istream is(&sb);
        unsigned char c = 'z';
        is >> c;
        assert( is.eof());
        assert( is.fail());
        assert(c == 'z');
    }
    {
        testbuf<char> sb("   abcdefghijk    ");
        std::istream is(&sb);
        unsigned char c;
        is >> c;
        assert(!is.eof());
        assert(!is.fail());
        assert(c == 'a');
        is >> c;
        assert(!is.eof());
        assert(!is.fail());
        assert(c == 'b');
        is >> c;
        assert(!is.eof());
        assert(!is.fail());
        assert(c == 'c');
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        testbuf<char> sb;
        std::basic_istream<char> is(&sb);
        is.exceptions(std::ios_base::failbit);

        bool threw = false;
        try {
            unsigned char n = 0;
            is >> n;
        } catch (std::ios_base::failure const&) {
            threw = true;
        }

        assert(!is.bad());
        assert(is.fail());
        assert(is.eof());
        assert(threw);
    }
    {
        testbuf<char> sb;
        std::basic_istream<char> is(&sb);
        is.exceptions(std::ios_base::eofbit);

        bool threw = false;
        try {
            unsigned char n = 0;
            is >> n;
        } catch (std::ios_base::failure const&) {
            threw = true;
        }

        assert(!is.bad());
        assert(is.fail());
        assert(is.eof());
        assert(threw);
    }
#endif // TEST_HAS_NO_EXCEPTIONS

    return 0;
}
