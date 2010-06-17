// -*- C++ -*-
//===-------------------------- algorithm ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class charT> struct regex_traits;

// regex_traits();

#include <regex>

int main()
{
    std::regex_traits<char> t1();
    std::regex_traits<wchar_t> t2();
}
