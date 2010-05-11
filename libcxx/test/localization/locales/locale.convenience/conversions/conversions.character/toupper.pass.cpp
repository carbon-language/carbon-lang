//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> charT toupper(charT c, const locale& loc);

#include <locale>
#include <cassert>

int main()
{
    std::locale l;
    assert(std::toupper(' ', l) == ' ');
    assert(std::toupper('<', l) == '<');
    assert(std::toupper('\x8', l) == '\x8');
    assert(std::toupper('A', l) == 'A');
    assert(std::toupper('a', l) == 'A');
    assert(std::toupper('z', l) == 'Z');
    assert(std::toupper('3', l) == '3');
    assert(std::toupper('.', l) == '.');
    assert(std::toupper('f', l) == 'F');
    assert(std::toupper('9', l) == '9');
    assert(std::toupper('+', l) == '+');
}
