//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <locale>

// template <> class ctype<char>

// static const mask* classic_table() throw();

#include <locale>
#include <cassert>

int main()
{
    typedef std::ctype<char> F;
    assert(F::classic_table() != 0);
}
