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

// template <class Value> any(Value &&)

// Test construction from a value.
// Concerns:
// ---------
// 1. The value is properly move/copied depending on the value category.
// 2. Both small and large values are properly handled.


#include <any>
#include <cassert>

#include "any_helpers.h"
#include "count_new.h"
#include "test_macros.h"

template <class Type>
void test_copy_value_throws()
{
#if !defined(TEST_HAS_NO_EXCEPTIONS)
    assert(Type::count == 0);
    {
        const Type t(42);
        assert(Type::count == 1);
        try {
            std::any a2 = t;
            assert(false);
        } catch (const my_any_exception&) {
            // do nothing
        } catch (...) {
            assert(false);
        }
        assert(Type::count == 1);
        assert(t.value == 42);
    }
    assert(Type::count == 0);
#endif
}

void test_move_value_throws()
{
#if !defined(TEST_HAS_NO_EXCEPTIONS)
    assert(throws_on_move::count == 0);
    {
        throws_on_move v;
        assert(throws_on_move::count == 1);
        try {
            std::any a = std::move(v);
            assert(false);
        } catch (const my_any_exception&) {
            // do nothing
        } catch (...) {
            assert(false);
        }
        assert(throws_on_move::count == 1);
    }
    assert(throws_on_move::count == 0);
#endif
}

template <class Type>
void test_copy_move_value() {
    // constructing from a small type should perform no allocations.
    DisableAllocationGuard g(isSmallType<Type>()); ((void)g);
    assert(Type::count == 0);
    Type::reset();
    {
        Type t(42);
        assert(Type::count == 1);

        std::any a = t;

        assert(Type::count == 2);
        assert(Type::copied == 1);
        assert(Type::moved == 0);
        assertContains<Type>(a, 42);
    }
    assert(Type::count == 0);
    Type::reset();
    {
        Type t(42);
        assert(Type::count == 1);

        std::any a = std::move(t);

        assert(Type::count == 2);
        assert(Type::copied == 0);
        assert(Type::moved == 1);
        assertContains<Type>(a, 42);
    }
}

// Test that any(ValueType&&) is *never* selected for a std::in_place_type_t specialization.
void test_sfinae_constraints() {
    using BadTag = std::in_place_type_t<int>;
    using OKTag = std::in_place_t;
    // Test that the tag type is properly handled in SFINAE
    BadTag t = std::in_place_type<int>;
    OKTag ot = std::in_place;
    {
        std::any a(t);
        assertContains<int>(a, 0);
    }
    {
        std::any a(std::move(t));
        assertContains<int>(a, 0);
    }
    {
        std::any a(ot);
        assert(containsType<OKTag>(a));
    }
    {
        struct Dummy { Dummy() = delete; };
        using T = std::in_place_type_t<Dummy>;
        static_assert(!std::is_constructible<std::any, T>::value, "");
    }
    {
        // Test that the ValueType&& constructor SFINAE's away when the
        // argument is non-copyable
        struct NoCopy {
          NoCopy() = default;
          NoCopy(NoCopy const&) = delete;
          NoCopy(int) {}
        };
        static_assert(!std::is_constructible<std::any, NoCopy>::value, "");
        static_assert(!std::is_constructible<std::any, NoCopy&>::value, "");
        static_assert(!std::is_convertible<NoCopy, std::any>::value, "");
    }
}

int main(int, char**) {
    test_copy_move_value<small>();
    test_copy_move_value<large>();
    test_copy_value_throws<small_throws_on_copy>();
    test_copy_value_throws<large_throws_on_copy>();
    test_move_value_throws();
    test_sfinae_constraints();

  return 0;
}
