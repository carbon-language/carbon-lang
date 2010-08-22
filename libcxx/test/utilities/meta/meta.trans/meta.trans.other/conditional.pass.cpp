//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// conditional

#include <type_traits>

int main()
{
    static_assert((std::is_same<std::conditional<true, char, int>::type, char>::value), "");
    static_assert((std::is_same<std::conditional<false, char, int>::type, int>::value), "");
}
