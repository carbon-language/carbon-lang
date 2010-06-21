//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class charT> struct regex_traits;

// locale_type getloc()const;

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
        std::locale::global(std::locale("en_US"));
        std::regex_traits<char> t1;
        assert(t1.getloc().name() == "en_US");
        std::regex_traits<wchar_t> t2;
        assert(t2.getloc().name() == "en_US");
    }
}
