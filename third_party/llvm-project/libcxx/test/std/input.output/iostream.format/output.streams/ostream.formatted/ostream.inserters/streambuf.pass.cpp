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

// basic_ostream<charT,traits>& operator<<(basic_streambuf<charT,traits>* sb);

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
    testbuf(const std::basic_string<CharT>& str)
        : str_(str)
    {
        base::setg(const_cast<CharT*>(str_.data()),
                   const_cast<CharT*>(str_.data()),
                   const_cast<CharT*>(str_.data() + str_.size()));
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
        testbuf<char> sb;
        std::ostream os(&sb);
        testbuf<char> sb2("testing...");
        assert(sb.str() == "");
        os << &sb2;
        assert(sb.str() == "testing...");
    }
    { // LWG 2221 - nullptr
        testbuf<char> sb;
        std::ostream os(&sb);
        os << nullptr;
        assert(sb.str().size() != 0);
        LIBCPP_ASSERT(sb.str() == "nullptr");
    }

  return 0;
}
