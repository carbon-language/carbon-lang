//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <regex>

// typedef basic_regex<char>    regex;

#include <regex>
#include <type_traits>

int main()
{
    static_assert((std::is_same<std::basic_regex<char>, std::regex>::value), "");
}
