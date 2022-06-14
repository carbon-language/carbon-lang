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

// any(any const &);

#include <any>
#include <cassert>

#include "any_helpers.h"
#include "count_new.h"
#include "test_macros.h"

template <class Type>
void test_copy_throws() {
#if !defined(TEST_HAS_NO_EXCEPTIONS)
    assert(Type::count == 0);
    {
        const std::any a = Type(42);
        assert(Type::count == 1);
        try {
            const std::any a2(a);
            assert(false);
        } catch (my_any_exception const &) {
            // do nothing
        } catch (...) {
            assert(false);
        }
        assert(Type::count == 1);
        assertContains<Type>(a, 42);
    }
    assert(Type::count == 0);
#endif
}

void test_copy_empty() {
    DisableAllocationGuard g; ((void)g); // No allocations should occur.
    std::any a1;
    std::any a2(a1);

    assertEmpty(a1);
    assertEmpty(a2);
}

template <class Type>
void test_copy()
{
    // Copying small types should not perform any allocations.
    DisableAllocationGuard g(isSmallType<Type>()); ((void)g);
    assert(Type::count == 0);
    Type::reset();
    {
        std::any a = Type(42);
        assert(Type::count == 1);
        assert(Type::copied == 0);

        std::any a2(a);

        assert(Type::copied == 1);
        assert(Type::count == 2);
        assertContains<Type>(a, 42);
        assertContains<Type>(a2, 42);

        // Modify a and check that a2 is unchanged
        modifyValue<Type>(a, -1);
        assertContains<Type>(a, -1);
        assertContains<Type>(a2, 42);

        // modify a2 and check that a is unchanged
        modifyValue<Type>(a2, 999);
        assertContains<Type>(a, -1);
        assertContains<Type>(a2, 999);

        // clear a and check that a2 is unchanged
        a.reset();
        assertEmpty(a);
        assertContains<Type>(a2, 999);
    }
    assert(Type::count == 0);
}

int main(int, char**) {
    test_copy<small>();
    test_copy<large>();
    test_copy_empty();
    test_copy_throws<small_throws_on_copy>();
    test_copy_throws<large_throws_on_copy>();

  return 0;
}
