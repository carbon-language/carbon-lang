//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <streambuf>

// template <class charT, class traits = char_traits<charT> >
// class basic_streambuf;

// basic_streambuf();

#include <streambuf>
#include <cassert>

template <class CharT>
struct test
    : public std::basic_streambuf<CharT>
{
    test()
    {
        assert(this->eback() == 0);
        assert(this->gptr() == 0);
        assert(this->egptr() == 0);
        assert(this->pbase() == 0);
        assert(this->pptr() == 0);
        assert(this->epptr() == 0);
    }
};

int main()
{
    {
        test<char> t;
        assert(t.getloc().name() == "C");
    }
    {
        test<wchar_t> t;
        assert(t.getloc().name() == "C");
    }
    std::locale::global(std::locale("en_US"));
    {
        test<char> t;
        assert(t.getloc().name() == "en_US");
    }
    {
        test<wchar_t> t;
        assert(t.getloc().name() == "en_US");
    }
}
