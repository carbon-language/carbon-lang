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

// template <class F, class T> constexpr decltype(auto) apply(F &&, T &&)

// Testing ref qualified functions

#include <experimental/tuple>
#include <cassert>

struct func_obj
{
    constexpr func_obj() {}

    constexpr int operator()() const & { return 1; }
    constexpr int operator()() const && { return 2; }
    constexpr int operator()() & { return 3; }
    constexpr int operator()() && { return 4; }
};

namespace ex = std::experimental;

int main()
{
    {
        constexpr func_obj f;
        constexpr std::tuple<> tp;

        static_assert(1 == ex::apply(static_cast<func_obj const &>(f), tp), "");
        static_assert(2 == ex::apply(static_cast<func_obj const &&>(f), tp), "");
    }
    {
        func_obj f;
        std::tuple<> tp;
        assert(1 == ex::apply(static_cast<func_obj const &>(f), tp));
        assert(2 == ex::apply(static_cast<func_obj const &&>(f), tp));
        assert(3 == ex::apply(static_cast<func_obj &>(f), tp));
        assert(4 == ex::apply(static_cast<func_obj &&>(f), tp));
    }
}
