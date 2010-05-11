//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <ios>

// typedef SZ_T streamsize;

#include <ios>
#include <type_traits>

int main()
{
    static_assert(std::is_integral<std::streamsize>::value, "");
    static_assert(std::is_signed<std::streamsize>::value, "");
}
