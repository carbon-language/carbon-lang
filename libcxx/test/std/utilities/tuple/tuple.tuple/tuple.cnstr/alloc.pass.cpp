//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <tuple>

// template <class... Types> class tuple;

// template <class Alloc>
//   explicit(see-below) tuple(allocator_arg_t, const Alloc& a);

#include <tuple>
#include <cassert>

#include "test_macros.h"
#include "DefaultOnly.h"
#include "allocators.h"
#include "../alloc_first.h"
#include "../alloc_last.h"

template <class T = void>
struct NonDefaultConstructible {
  constexpr NonDefaultConstructible() {
      static_assert(!std::is_same<T, T>::value, "Default Ctor instantiated");
  }

  explicit constexpr NonDefaultConstructible(int) {}
};


struct DerivedFromAllocArgT : std::allocator_arg_t {};

int main(int, char**)
{
    {
        std::tuple<> t(std::allocator_arg, A1<int>());
    }
    {
        std::tuple<int> t(std::allocator_arg, A1<int>());
        assert(std::get<0>(t) == 0);
    }
    {
        std::tuple<DefaultOnly> t(std::allocator_arg, A1<int>());
        assert(std::get<0>(t) == DefaultOnly());
    }
    {
        assert(!alloc_first::allocator_constructed);
        std::tuple<alloc_first> t(std::allocator_arg, A1<int>(5));
        assert(alloc_first::allocator_constructed);
        assert(std::get<0>(t) == alloc_first());
    }
    {
        assert(!alloc_last::allocator_constructed);
        std::tuple<alloc_last> t(std::allocator_arg, A1<int>(5));
        assert(alloc_last::allocator_constructed);
        assert(std::get<0>(t) == alloc_last());
    }
    {
        alloc_first::allocator_constructed = false;
        std::tuple<DefaultOnly, alloc_first> t(std::allocator_arg, A1<int>(5));
        assert(std::get<0>(t) == DefaultOnly());
        assert(alloc_first::allocator_constructed);
        assert(std::get<1>(t) == alloc_first());
    }
    {
        alloc_first::allocator_constructed = false;
        alloc_last::allocator_constructed = false;
        std::tuple<DefaultOnly, alloc_first, alloc_last> t(std::allocator_arg,
                                                           A1<int>(5));
        assert(std::get<0>(t) == DefaultOnly());
        assert(alloc_first::allocator_constructed);
        assert(std::get<1>(t) == alloc_first());
        assert(alloc_last::allocator_constructed);
        assert(std::get<2>(t) == alloc_last());
    }
    {
        alloc_first::allocator_constructed = false;
        alloc_last::allocator_constructed = false;
        std::tuple<DefaultOnly, alloc_first, alloc_last> t(std::allocator_arg,
                                                           A2<int>(5));
        assert(std::get<0>(t) == DefaultOnly());
        assert(!alloc_first::allocator_constructed);
        assert(std::get<1>(t) == alloc_first());
        assert(!alloc_last::allocator_constructed);
        assert(std::get<2>(t) == alloc_last());
    }
    {
        // Test that we can use a tag derived from allocator_arg_t
        struct DerivedFromAllocatorArgT : std::allocator_arg_t { };
        DerivedFromAllocatorArgT derived;
        std::tuple<> t1(derived, A1<int>());
        std::tuple<int> t2(derived, A1<int>());
        std::tuple<int, int> t3(derived, A1<int>());
    }
    {
        // Test that the uses-allocator default constructor does not evaluate
        // its SFINAE when it otherwise shouldn't be selected. Do this by
        // using 'NonDefaultConstructible' which will cause a compile error
        // if std::is_default_constructible is evaluated on it.
        using T = NonDefaultConstructible<>;
        T v(42);
        std::tuple<T, T> t(v, v);
        (void)t;
        std::tuple<T, T> t2(42, 42);
        (void)t2;
    }

  return 0;
}
