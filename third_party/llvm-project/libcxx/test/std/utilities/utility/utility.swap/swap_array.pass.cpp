//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template<ValueType T, size_t N>
//   requires Swappable<T>
//   void
//   swap(T (&a)[N], T (&b)[N]);

#include <algorithm>
#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>

#include "test_macros.h"


#if TEST_STD_VER >= 11
struct CopyOnly {
    CopyOnly() {}
    CopyOnly(CopyOnly const&) noexcept {}
    CopyOnly& operator=(CopyOnly const&) { return *this; }
};


struct NoexceptMoveOnly {
    NoexceptMoveOnly() {}
    NoexceptMoveOnly(NoexceptMoveOnly&&) noexcept {}
    NoexceptMoveOnly& operator=(NoexceptMoveOnly&&) noexcept { return *this; }
};

struct NotMoveConstructible {
    NotMoveConstructible() {}
    NotMoveConstructible& operator=(NotMoveConstructible&&) { return *this; }
private:
    NotMoveConstructible(NotMoveConstructible&&);
};

template <class Tp>
auto can_swap_test(int) -> decltype(std::swap(std::declval<Tp>(), std::declval<Tp>()));

template <class Tp>
auto can_swap_test(...) -> std::false_type;

template <class Tp>
constexpr bool can_swap() {
    return std::is_same<decltype(can_swap_test<Tp>(0)), void>::value;
}
#endif

#if TEST_STD_VER > 17
constexpr bool test_swap_constexpr()
{
    int i[3] = {1, 2, 3};
    int j[3] = {4, 5, 6};
    std::swap(i, j);
    return i[0] == 4 &&
           i[1] == 5 &&
           i[2] == 6 &&
           j[0] == 1 &&
           j[1] == 2 &&
           j[2] == 3;
}
#endif // TEST_STD_VER > 17

int main(int, char**)
{
    {
        int i[3] = {1, 2, 3};
        int j[3] = {4, 5, 6};
        std::swap(i, j);
        assert(i[0] == 4);
        assert(i[1] == 5);
        assert(i[2] == 6);
        assert(j[0] == 1);
        assert(j[1] == 2);
        assert(j[2] == 3);
    }
#if TEST_STD_VER >= 11
    {
        std::unique_ptr<int> i[3];
        for (int k = 0; k < 3; ++k)
            i[k].reset(new int(k+1));
        std::unique_ptr<int> j[3];
        for (int k = 0; k < 3; ++k)
            j[k].reset(new int(k+4));
        std::swap(i, j);
        assert(*i[0] == 4);
        assert(*i[1] == 5);
        assert(*i[2] == 6);
        assert(*j[0] == 1);
        assert(*j[1] == 2);
        assert(*j[2] == 3);
    }
    {
        using CA = CopyOnly[42];
        using MA = NoexceptMoveOnly[42];
        using NA = NotMoveConstructible[42];
        static_assert(can_swap<CA&>(), "");
        static_assert(can_swap<MA&>(), "");
        static_assert(!can_swap<NA&>(), "");

        CA ca;
        MA ma;
        static_assert(!noexcept(std::swap(ca, ca)), "");
        static_assert(noexcept(std::swap(ma, ma)), "");
    }
#endif

#if TEST_STD_VER > 17
    static_assert(test_swap_constexpr());
#endif // TEST_STD_VER > 17

  return 0;
}
