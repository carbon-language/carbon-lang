//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// tuple_size<pair<T1, T2> >::value

#include <utility>

int main()
{
    {
        typedef std::pair<int, short> P1;
        static_assert((std::tuple_size<P1>::value == 2), "");
    }
}
