//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <ios>

// template <class charT, class traits> class basic_ios

// bool operator!() const;

#include <ios>
#include <cassert>

int main()
{
    std::ios ios(0);
    assert(!ios == ios.fail());
    ios.setstate(std::ios::failbit);
    assert(!ios == ios.fail());
}
