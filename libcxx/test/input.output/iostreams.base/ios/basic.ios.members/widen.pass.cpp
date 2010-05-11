//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <ios>

// template <class charT, class traits> class basic_ios

// char_type widen(char c) const;

#include <ios>
#include <cassert>

int main()
{
    const std::ios ios(0);
    assert(ios.widen('c') == 'c');
}
