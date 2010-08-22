//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// enable_if

#include <type_traits>

int main()
{
    static_assert((std::is_same<std::enable_if<true>::type, void>::value), "");
    static_assert((std::is_same<std::enable_if<true, int>::type, int>::value), "");
}
