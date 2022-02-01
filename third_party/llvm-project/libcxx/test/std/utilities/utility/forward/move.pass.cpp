//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test move

#include <utility>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

class move_only
{
    move_only(const move_only&);
    move_only& operator=(const move_only&);
public:
    move_only(move_only&&) {}
    move_only& operator=(move_only&&) {return *this;}

    move_only() {}
};

move_only source() {return move_only();}
const move_only csource() {return move_only();}

void test(move_only) {}

int x = 42;
const int& cx = x;

template <class QualInt>
QualInt get() TEST_NOEXCEPT { return static_cast<QualInt>(x); }


int copy_ctor = 0;
int move_ctor = 0;

struct A {
    A() {}
    A(const A&) {++copy_ctor;}
    A(A&&) {++move_ctor;}
    A& operator=(const A&) = delete;
};

#if TEST_STD_VER > 11
constexpr bool test_constexpr_move() {
    int y = 42;
    const int cy = y;
    return std::move(y) == 42
        && std::move(cy) == 42
        && std::move(static_cast<int&&>(y)) == 42
        && std::move(static_cast<int const&&>(y)) == 42;
}
#endif
int main(int, char**)
{
    { // Test return type and noexcept.
        static_assert(std::is_same<decltype(std::move(x)), int&&>::value, "");
        ASSERT_NOEXCEPT(std::move(x));
        static_assert(std::is_same<decltype(std::move(cx)), const int&&>::value, "");
        ASSERT_NOEXCEPT(std::move(cx));
        static_assert(std::is_same<decltype(std::move(42)), int&&>::value, "");
        ASSERT_NOEXCEPT(std::move(42));
        static_assert(std::is_same<decltype(std::move(get<const int&&>())), const int&&>::value, "");
        ASSERT_NOEXCEPT(std::move(get<int const&&>()));
    }
    { // test copy and move semantics
        A a;
        const A ca = A();

        assert(copy_ctor == 0);
        assert(move_ctor == 0);

        A a2 = a; (void)a2;
        assert(copy_ctor == 1);
        assert(move_ctor == 0);

        A a3 = std::move(a); (void)a3;
        assert(copy_ctor == 1);
        assert(move_ctor == 1);

        A a4 = ca; (void)a4;
        assert(copy_ctor == 2);
        assert(move_ctor == 1);

        A a5 = std::move(ca); (void)a5;
        assert(copy_ctor == 3);
        assert(move_ctor == 1);
    }
    { // test on a move only type
        move_only mo;
        test(std::move(mo));
        test(source());
    }
#if TEST_STD_VER > 11
    {
        constexpr int y = 42;
        static_assert(std::move(y) == 42, "");
        static_assert(test_constexpr_move(), "");
    }
#endif
#if TEST_STD_VER == 11 && defined(_LIBCPP_VERSION)
    // Test that std::forward is constexpr in C++11. This is an extension
    // provided by both libc++ and libstdc++.
    {
        constexpr int y = 42;
        static_assert(std::move(y) == 42, "");
    }
#endif

  return 0;
}
