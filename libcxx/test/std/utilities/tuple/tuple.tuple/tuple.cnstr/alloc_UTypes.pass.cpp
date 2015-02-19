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

// template <class Alloc, class... UTypes>
//   tuple(allocator_arg_t, const Alloc& a, UTypes&&...);

// UNSUPPORTED: c++98, c++03

#include <tuple>
#include <cassert>

#include "MoveOnly.h"
#include "allocators.h"
#include "../alloc_first.h"
#include "../alloc_last.h"

struct NoDefault { NoDefault() = delete; };

// Make sure the _Up... constructor SFINAEs out when the types that
// are not explicitly initialized are not all default constructible.
// Otherwise, std::is_constructible would return true but instantiating
// the constructor would fail.
void test_default_constructible_extension_sfinae()
{
    {
        typedef std::tuple<MoveOnly, NoDefault> Tuple;

        static_assert(!std::is_constructible<
            Tuple,
            std::allocator_arg_t, A1<int>, MoveOnly
        >::value, "");

        static_assert(std::is_constructible<
            Tuple,
            std::allocator_arg_t, A1<int>, MoveOnly, NoDefault
        >::value, "");
    }
    {
        typedef std::tuple<MoveOnly, MoveOnly, NoDefault> Tuple;

        static_assert(!std::is_constructible<
            Tuple,
            std::allocator_arg_t, A1<int>, MoveOnly, MoveOnly
        >::value, "");

        static_assert(std::is_constructible<
            Tuple,
            std::allocator_arg_t, A1<int>, MoveOnly, MoveOnly, NoDefault
        >::value, "");
    }
    {
        // Same idea as above but with a nested tuple
        typedef std::tuple<MoveOnly, NoDefault> Tuple;
        typedef std::tuple<MoveOnly, Tuple, MoveOnly, MoveOnly> NestedTuple;

        static_assert(!std::is_constructible<
            NestedTuple,
            std::allocator_arg_t, A1<int>, MoveOnly, MoveOnly, MoveOnly, MoveOnly
        >::value, "");

        static_assert(std::is_constructible<
            NestedTuple,
            std::allocator_arg_t, A1<int>, MoveOnly, Tuple, MoveOnly, MoveOnly
        >::value, "");
    }
    {
        typedef std::tuple<MoveOnly, int> Tuple;
        typedef std::tuple<MoveOnly, Tuple, MoveOnly, MoveOnly> NestedTuple;

        static_assert(std::is_constructible<
            NestedTuple,
            std::allocator_arg_t, A1<int>, MoveOnly, MoveOnly, MoveOnly, MoveOnly
        >::value, "");

        static_assert(std::is_constructible<
            NestedTuple,
            std::allocator_arg_t, A1<int>, MoveOnly, Tuple, MoveOnly, MoveOnly
        >::value, "");
    }
}

int main()
{
    {
        std::tuple<MoveOnly> t(std::allocator_arg, A1<int>(), MoveOnly(0));
        assert(std::get<0>(t) == 0);
    }
    {
        std::tuple<MoveOnly, MoveOnly> t(std::allocator_arg, A1<int>(),
                                         MoveOnly(0), MoveOnly(1));
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == 1);
    }
    {
        std::tuple<MoveOnly, MoveOnly, MoveOnly> t(std::allocator_arg, A1<int>(),
                                                   MoveOnly(0),
                                                   1, 2);
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == 1);
        assert(std::get<2>(t) == 2);
    }
    {
        alloc_first::allocator_constructed = false;
        alloc_last::allocator_constructed = false;
        std::tuple<int, alloc_first, alloc_last> t(std::allocator_arg,
                                                   A1<int>(5), 1, 2, 3);
        assert(std::get<0>(t) == 1);
        assert(alloc_first::allocator_constructed);
        assert(std::get<1>(t) == alloc_first(2));
        assert(alloc_last::allocator_constructed);
        assert(std::get<2>(t) == alloc_last(3));
    }
    // extensions
    {
        std::tuple<MoveOnly, MoveOnly, MoveOnly> t(std::allocator_arg, A1<int>(),
                                                   0, 1);
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == 1);
        assert(std::get<2>(t) == MoveOnly());
    }
    {
        std::tuple<MoveOnly, MoveOnly, MoveOnly> t(std::allocator_arg, A1<int>(),
                                                   0);
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == MoveOnly());
        assert(std::get<2>(t) == MoveOnly());
    }
    // Check that SFINAE is properly applied with the default reduced arity
    // constructor extensions.
    test_default_constructible_extension_sfinae();
}
