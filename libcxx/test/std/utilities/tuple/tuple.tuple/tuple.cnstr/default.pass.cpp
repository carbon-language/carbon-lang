//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// constexpr tuple();

#include <tuple>
#include <string>
#include <cassert>
#include <type_traits>

#include "DefaultOnly.h"

struct NoDefault {
    NoDefault() = delete;
    explicit NoDefault(int) { }
};

struct NoExceptDefault {
    NoExceptDefault() noexcept = default;
};

struct ThrowingDefault {
    ThrowingDefault() { }
};

int main()
{
    {
        std::tuple<> t;
    }
    {
        std::tuple<int> t;
        assert(std::get<0>(t) == 0);
    }
    {
        std::tuple<int, char*> t;
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == nullptr);
    }
    {
        std::tuple<int, char*, std::string> t;
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == nullptr);
        assert(std::get<2>(t) == "");
    }
    {
        std::tuple<int, char*, std::string, DefaultOnly> t;
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == nullptr);
        assert(std::get<2>(t) == "");
        assert(std::get<3>(t) == DefaultOnly());
    }
    {
        // See bug #21157.
        static_assert(!std::is_default_constructible<std::tuple<NoDefault>>(), "");
        static_assert(!std::is_default_constructible<std::tuple<DefaultOnly, NoDefault>>(), "");
        static_assert(!std::is_default_constructible<std::tuple<NoDefault, DefaultOnly, NoDefault>>(), "");
    }
    {
        static_assert(noexcept(std::tuple<NoExceptDefault>()), "");
        static_assert(noexcept(std::tuple<NoExceptDefault, NoExceptDefault>()), "");

        static_assert(!noexcept(std::tuple<ThrowingDefault, NoExceptDefault>()), "");
        static_assert(!noexcept(std::tuple<NoExceptDefault, ThrowingDefault>()), "");
        static_assert(!noexcept(std::tuple<ThrowingDefault, ThrowingDefault>()), "");
    }
#ifndef _LIBCPP_HAS_NO_CONSTEXPR
    {
        constexpr std::tuple<> t;
    }
    {
        constexpr std::tuple<int> t;
        assert(std::get<0>(t) == 0);
    }
    {
        constexpr std::tuple<int, char*> t;
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == nullptr);
    }
#endif
}
