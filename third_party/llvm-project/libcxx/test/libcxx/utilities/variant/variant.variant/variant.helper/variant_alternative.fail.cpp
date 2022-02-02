//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <variant>
// UNSUPPORTED: c++03, c++11, c++14


// template <size_t I, class T> struct variant_alternative; // undefined
// template <size_t I, class T> struct variant_alternative<I, const T>;
// template <size_t I, class T> struct variant_alternative<I, volatile T>;
// template <size_t I, class T> struct variant_alternative<I, const volatile T>;
// template <size_t I, class T>
//   using variant_alternative_t = typename variant_alternative<I, T>::type;
//
// template <size_t I, class... Types>
//    struct variant_alternative<I, variant<Types...>>;


#include <variant>
#include <cassert>


int main(int, char**)
{
    {
        typedef std::variant<int, double> T;
        std::variant_alternative<2, T>::type foo; // expected-note {{requested here}}
        // expected-error-re@variant:* {{static_assert failed{{( due to requirement '2U[L]{0,2} < sizeof...\(_Types\)')?}} "Index out of bounds in std::variant_alternative<>"}}
    }

  return 0;
}
