//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <regex>

// typedef sub_match<const char*>   csub_match;

#include <regex>
#include <type_traits>

int main()
{
    static_assert((std::is_same<std::sub_match<const char*>, std::csub_match>::value), "");
}
