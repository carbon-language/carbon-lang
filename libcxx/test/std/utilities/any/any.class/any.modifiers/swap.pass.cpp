//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Throwing bad_any_cast is supported starting in macosx10.13
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}} && !no-exceptions

// <any>

// any::swap(any &) noexcept

// Test swap(large, small) and swap(small, large)

#include <any>
#include <cassert>

#include "test_macros.h"
#include "any_helpers.h"

template <class LHS, class RHS>
void test_swap() {
    assert(LHS::count == 0);
    assert(RHS::count == 0);
    {
        std::any a1 = LHS(1);
        std::any a2 = RHS(2);
        assert(LHS::count == 1);
        assert(RHS::count == 1);

        a1.swap(a2);

        assert(LHS::count == 1);
        assert(RHS::count == 1);

        assertContains<RHS>(a1, 2);
        assertContains<LHS>(a2, 1);
    }
    assert(LHS::count == 0);
    assert(RHS::count == 0);
    assert(LHS::copied == 0);
    assert(RHS::copied == 0);
}

template <class Tp>
void test_swap_empty() {
    assert(Tp::count == 0);
    {
        std::any a1 = Tp(1);
        std::any a2;
        assert(Tp::count == 1);

        a1.swap(a2);

        assert(Tp::count == 1);

        assertContains<Tp>(a2, 1);
        assertEmpty(a1);
    }
    assert(Tp::count == 0);
    {
        std::any a1 = Tp(1);
        std::any a2;
        assert(Tp::count == 1);

        a2.swap(a1);

        assert(Tp::count == 1);

        assertContains<Tp>(a2, 1);
        assertEmpty(a1);
    }
    assert(Tp::count == 0);
    assert(Tp::copied == 0);
}

void test_noexcept()
{
    std::any a1;
    std::any a2;
    ASSERT_NOEXCEPT(a1.swap(a2));
}

void test_self_swap() {
    {
        // empty
        std::any a;
        a.swap(a);
        assertEmpty(a);
    }
    { // small
        using T = small;
        std::any a = T(42);
        T::reset();
        a.swap(a);
        assertContains<T>(a, 42);
        assert(T::count == 1);
        assert(T::copied == 0);
        LIBCPP_ASSERT(T::moved == 0);
    }
    assert(small::count == 0);
    { // large
        using T = large;
        std::any a = T(42);
        T::reset();
        a.swap(a);
        assertContains<T>(a, 42);
        assert(T::count == 1);
        assert(T::copied == 0);
        LIBCPP_ASSERT(T::moved == 0);
    }
    assert(large::count == 0);
}

int main(int, char**)
{
    test_noexcept();
    test_swap_empty<small>();
    test_swap_empty<large>();
    test_swap<small1, small2>();
    test_swap<large1, large2>();
    test_swap<small, large>();
    test_swap<large, small>();
    test_self_swap();

  return 0;
}
