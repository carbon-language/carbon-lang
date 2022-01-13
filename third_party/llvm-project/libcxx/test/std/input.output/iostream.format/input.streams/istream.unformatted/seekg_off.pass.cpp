//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11}}

// <istream>

// basic_istream<charT,traits>& seekg(off_type off, ios_base::seekdir dir);

#include <istream>
#include <cassert>

#include "test_macros.h"

int seekoff_called = 0;

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
protected:
    typename base::pos_type seekoff(typename base::off_type off,
                                    std::ios_base::seekdir,
                                    std::ios_base::openmode which)
    {
        assert(which == std::ios_base::in);
        ++seekoff_called;
        return off;
    }
};

int main(int, char**)
{
    {
        testbuf<char> sb(" 123456789");
        std::istream is(&sb);
        is.seekg(5, std::ios_base::cur);
        assert(is.good());
        assert(seekoff_called == 1);
        is.seekg(-1, std::ios_base::beg);
        assert(is.fail());
        assert(seekoff_called == 2);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        testbuf<wchar_t> sb(L" 123456789");
        std::wistream is(&sb);
        is.seekg(5, std::ios_base::cur);
        assert(is.good());
        assert(seekoff_called == 3);
        is.seekg(-1, std::ios_base::beg);
        assert(is.fail());
        assert(seekoff_called == 4);
    }
#endif
    {
        testbuf<char> sb(" 123456789");
        std::istream is(&sb);
        is.setstate(std::ios_base::eofbit);
        assert(is.eof());
        is.seekg(5, std::ios_base::beg);
        assert(is.good());
        assert(!is.eof());
    }

  return 0;
}
