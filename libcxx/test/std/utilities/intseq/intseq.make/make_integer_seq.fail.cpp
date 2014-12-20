//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template<class T, T N>
//   using make_integer_sequence = integer_sequence<T, 0, 1, ..., N-1>;

#include <utility>
#include <type_traits>
#include <cassert>

int main()
{
#if _LIBCPP_STD_VER > 11

    std::make_integer_sequence<int, -3>::value_type i;

#else

X

#endif  // _LIBCPP_STD_VER > 11
}
