//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class charT, class traits = regex_traits<charT>> class basic_regex;

// locale_type imbue(locale_type loc);

#include <regex>
#include <locale>
#include <cassert>

int main()
{
    std::regex r;
    std::locale loc = r.imbue(std::locale("en_US"));
    assert(loc.name() == "C");
    assert(r.getloc().name() == "en_US");
    loc = r.imbue(std::locale("C"));
    assert(loc.name() == "en_US");
    assert(r.getloc().name() == "C");
}
