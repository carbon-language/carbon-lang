//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<wchar_t>

// static constexpr char_type to_char_type(int_type c);

#include <string>
#include <cassert>

int main()
{
    assert(std::char_traits<wchar_t>::to_char_type(L'a') == L'a');
    assert(std::char_traits<wchar_t>::to_char_type(L'A') == L'A');
    assert(std::char_traits<wchar_t>::to_char_type(0) == 0);
}
