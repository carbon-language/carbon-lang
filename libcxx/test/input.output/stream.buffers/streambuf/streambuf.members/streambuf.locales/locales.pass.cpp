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

// locale pubimbue(const locale& loc);
// locale getloc() const;

#include <streambuf>
#include <cassert>

template <class CharT>
struct test
    : public std::basic_streambuf<CharT>
{
    test() {}

    void imbue(const std::locale&)
    {
        assert(this->getloc().name() == "en_US");
    }
};

int main()
{
    {
        test<char> t;
        assert(t.getloc().name() == "C");
    }
    std::locale::global(std::locale("en_US"));
    {
        test<char> t;
        assert(t.getloc().name() == "en_US");
        assert(t.pubimbue(std::locale("fr_FR")).name() == "en_US");
        assert(t.getloc().name() == "fr_FR");
    }
}
