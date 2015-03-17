//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <experimental/tuple>

// template <class T> constexpr size_t tuple_size_v = tuple_size<T>::value;

#include <experimental/tuple>
#include <utility>
#include <array>

namespace ex = std::experimental;

template <class Tuple, int Expect>
void test()
{
    static_assert(ex::tuple_size_v<Tuple> == Expect, "");
    static_assert(ex::tuple_size_v<Tuple> == std::tuple_size<Tuple>::value, "");
    static_assert(ex::tuple_size_v<Tuple const> == std::tuple_size<Tuple>::value, "");
    static_assert(ex::tuple_size_v<Tuple volatile> == std::tuple_size<Tuple>::value, "");
    static_assert(ex::tuple_size_v<Tuple const volatile> == std::tuple_size<Tuple>::value, "");
}

int main()
{
    test<std::tuple<>, 0>();

    test<std::tuple<int>, 1>();
    test<std::array<int, 1>, 1>();

    test<std::tuple<int, int>, 2>();
    test<std::pair<int, int>, 2>();
    test<std::array<int, 2>, 2>();

    test<std::tuple<int, int, int>, 3>();
    test<std::array<int, 3>, 3>();
}
