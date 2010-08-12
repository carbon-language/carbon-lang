//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <regex>

// typedef match_results<const wchar_t*>   wcmatch;

#include <regex>
#include <type_traits>

int main()
{
    static_assert((std::is_same<std::match_results<const wchar_t*>, std::wcmatch>::value), "");
}
