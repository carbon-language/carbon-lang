//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class Alloc>
//   tuple(allocator_arg_t, const Alloc& a, const tuple&);

// UNSUPPORTED: c++03

#include <tuple>
#include <cassert>

#include "test_macros.h"
#include "allocators.h"
#include "../alloc_first.h"
#include "../alloc_last.h"

int main(int, char**)
{
    {
        typedef std::tuple<> T;
        T t0;
        T t(std::allocator_arg, A1<int>(), t0);
    }
    {
        typedef std::tuple<int> T;
        T t0(2);
        T t(std::allocator_arg, A1<int>(), t0);
        assert(std::get<0>(t) == 2);
    }
    {
        typedef std::tuple<alloc_first> T;
        T t0(2);
        alloc_first::allocator_constructed = false;
        T t(std::allocator_arg, A1<int>(5), t0);
        assert(alloc_first::allocator_constructed);
        assert(std::get<0>(t) == 2);
    }
    {
        typedef std::tuple<alloc_last> T;
        T t0(2);
        alloc_last::allocator_constructed = false;
        T t(std::allocator_arg, A1<int>(5), t0);
        assert(alloc_last::allocator_constructed);
        assert(std::get<0>(t) == 2);
    }
    {
        typedef std::tuple<alloc_first, alloc_last> T;
        T t0(2, 3);
        alloc_first::allocator_constructed = false;
        alloc_last::allocator_constructed = false;
        T t(std::allocator_arg, A1<int>(5), t0);
        assert(alloc_first::allocator_constructed);
        assert(alloc_last::allocator_constructed);
        assert(std::get<0>(t) == 2);
        assert(std::get<1>(t) == 3);
    }
    {
        typedef std::tuple<int, alloc_first, alloc_last> T;
        T t0(1, 2, 3);
        alloc_first::allocator_constructed = false;
        alloc_last::allocator_constructed = false;
        T t(std::allocator_arg, A1<int>(5), t0);
        assert(alloc_first::allocator_constructed);
        assert(alloc_last::allocator_constructed);
        assert(std::get<0>(t) == 1);
        assert(std::get<1>(t) == 2);
        assert(std::get<2>(t) == 3);
    }
    {
        // Test that we can use a tag derived from allocator_arg_t
        struct DerivedFromAllocatorArgT : std::allocator_arg_t { };
        DerivedFromAllocatorArgT derived;
        std::tuple<int> from(3);
        std::tuple<int> t0(derived, A1<int>(), from);
    }

    return 0;
}
