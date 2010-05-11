//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<char>

// static void assign(char_type& c1, const char_type& c2);

#include <string>
#include <cassert>

int main()
{
    char c = '\0';
    std::char_traits<char>::assign(c, 'a');
    assert(c == 'a');
}
