//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <regex>

// typedef match_results<wstring::const_iterator>   wsmatch;

#include <regex>
#include <type_traits>

int main()
{
    static_assert((std::is_same<std::match_results<std::wstring::const_iterator>, std::wsmatch>::value), "");
}
