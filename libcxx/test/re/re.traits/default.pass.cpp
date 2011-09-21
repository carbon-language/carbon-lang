// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class charT> struct regex_traits;

// regex_traits();

#include <regex>
#include <cassert>

int main()
{
    {
        std::regex_traits<char> t1;
        assert(t1.getloc().name() == "C");
        std::regex_traits<wchar_t> t2;
        assert(t2.getloc().name() == "C");
    }
    {
        std::locale::global(std::locale("en_US.UTF-8"));
        std::regex_traits<char> t1;
        assert(t1.getloc().name() == "en_US.UTF-8");
        std::regex_traits<wchar_t> t2;
        assert(t2.getloc().name() == "en_US.UTF-8");
    }
}
