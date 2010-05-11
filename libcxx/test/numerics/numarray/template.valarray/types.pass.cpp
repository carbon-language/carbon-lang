//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T>
// class valarray
// {
// public:
//     typedef T value_type;
//     ...

#include <valarray>
#include <type_traits>

int main()
{
    static_assert((std::is_same<std::valarray<int>::value_type, int>::value), "");
    static_assert((std::is_same<std::valarray<double>::value_type, double>::value), "");
}
