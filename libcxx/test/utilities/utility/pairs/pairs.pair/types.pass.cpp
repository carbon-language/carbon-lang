//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2>
// struct pair
// {
//     typedef T1 first_type;
//     typedef T2 second_type;


#include <utility>
#include <type_traits>

int main()
{
    typedef std::pair<float, short*> P;
    static_assert((std::is_same<P::first_type, float>::value), "");
    static_assert((std::is_same<P::second_type, short*>::value), "");
}
