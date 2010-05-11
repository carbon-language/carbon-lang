//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <array>

// tuple_size<array<T, N> >::value

#include <array>

int main()
{
    {
        typedef double T;
        typedef std::array<T, 3> C;
        static_assert((std::tuple_size<C>::value == 3), "");
    }
    {
        typedef double T;
        typedef std::array<T, 0> C;
        static_assert((std::tuple_size<C>::value == 0), "");
    }
}
