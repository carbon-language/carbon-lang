//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// has_copy_assign

#include <type_traits>

int main()
{
    static_assert((std::has_copy_assign<int>::value), "");
}
