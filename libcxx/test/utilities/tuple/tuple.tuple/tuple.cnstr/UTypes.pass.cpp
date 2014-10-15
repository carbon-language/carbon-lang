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

// template <class... UTypes>
//   explicit tuple(UTypes&&... u);

#include <tuple>
#include <cassert>
#include <type_traits>

#include "../MoveOnly.h"

#if _LIBCPP_STD_VER > 11

struct Empty {};
struct A
{
    int id_;
    explicit constexpr A(int i) : id_(i) {}
};

#endif

struct NoDefault { NoDefault() = delete; };

int main()
{
    {
        std::tuple<MoveOnly> t(MoveOnly(0));
        assert(std::get<0>(t) == 0);
    }
    {
        std::tuple<MoveOnly, MoveOnly> t(MoveOnly(0), MoveOnly(1));
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == 1);
    }
    {
        std::tuple<MoveOnly, MoveOnly, MoveOnly> t(MoveOnly(0),
                                                   MoveOnly(1),
                                                   MoveOnly(2));
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == 1);
        assert(std::get<2>(t) == 2);
    }
    // extensions
    {
        std::tuple<MoveOnly, MoveOnly, MoveOnly> t(MoveOnly(0),
                                                   MoveOnly(1));
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == 1);
        assert(std::get<2>(t) == MoveOnly());
    }
    {
        std::tuple<MoveOnly, MoveOnly, MoveOnly> t(MoveOnly(0));
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == MoveOnly());
        assert(std::get<2>(t) == MoveOnly());
    }
    {
        // Make sure the _Up... constructor SFINAEs out when the types that
        // are not explicitly initialized are not all default constructible.
        // Otherwise, std::is_constructible would return true but instantiating
        // the constructor would fail.
        static_assert(!std::is_constructible<std::tuple<MoveOnly, NoDefault>, MoveOnly>(), "");
        static_assert(!std::is_constructible<std::tuple<MoveOnly, MoveOnly, NoDefault>, MoveOnly, MoveOnly>(), "");
    }
#if _LIBCPP_STD_VER > 11
    {
        constexpr std::tuple<Empty> t0{Empty()};
    }
    {
        constexpr std::tuple<A, A> t(3, 2);
        static_assert(std::get<0>(t).id_ == 3, "");
    }
#endif
}
