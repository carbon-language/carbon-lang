//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <regex>

// typedef sub_match<string::const_iterator>   ssub_match;

#include <regex>
#include <type_traits>

int main()
{
    static_assert((std::is_same<std::sub_match<std::string::const_iterator>, std::ssub_match>::value), "");
}
