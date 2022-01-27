//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ostream>

// template <class charT, class traits = char_traits<charT> >
//   class basic_ostream;

// template<class traits>
//   basic_ostream<char,traits>& operator<<(basic_ostream<char,traits>& out, const signed char* s);

#include <ostream>
#include <cassert>

#include "test_macros.h"

template <class CharT>
class testbuf
    : public std::basic_streambuf<CharT>
{
    typedef std::basic_streambuf<CharT> base;
    std::basic_string<CharT> str_;
public:
    testbuf()
    {
    }

    std::basic_string<CharT> str() const
        {return std::basic_string<CharT>(base::pbase(), base::pptr());}

protected:

    virtual typename base::int_type
        overflow(typename base::int_type ch = base::traits_type::eof())
        {
            if (ch != base::traits_type::eof())
            {
                int n = static_cast<int>(str_.size());
                str_.push_back(static_cast<CharT>(ch));
                str_.resize(str_.capacity());
                base::setp(const_cast<CharT*>(str_.data()),
                           const_cast<CharT*>(str_.data() + str_.size()));
                base::pbump(n+1);
            }
            return ch;
        }
};

int main(int, char**)
{
    {
        std::ostream os((std::streambuf*)0);
        const signed char* c = (const signed char*)"123";
        os << c;
        assert(os.bad());
        assert(os.fail());
    }
    {
        testbuf<char> sb;
        std::ostream os(&sb);
        const signed char* c = (const signed char*)"123";
        os << c;
        assert(sb.str() == "123");
    }
    {
        testbuf<char> sb;
        std::ostream os(&sb);
        os.width(5);
        const signed char* c = (const signed char*)"123";
        os << c;
        assert(sb.str() == "  123");
        assert(os.width() == 0);
    }
    {
        testbuf<char> sb;
        std::ostream os(&sb);
        os.width(5);
        left(os);
        const signed char* c = (const signed char*)"123";
        os << c;
        assert(sb.str() == "123  ");
        assert(os.width() == 0);
    }

  return 0;
}
