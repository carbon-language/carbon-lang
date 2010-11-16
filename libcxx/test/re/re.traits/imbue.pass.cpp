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

// locale_type imbue(locale_type l);

#include <regex>
#include <locale>
#include <cassert>

int main()
{
    {
        std::regex_traits<char> t;
        std::locale loc = t.imbue(std::locale("en_US"));
        assert(loc.name() == "C");
        assert(t.getloc().name() == "en_US");
    }
}
