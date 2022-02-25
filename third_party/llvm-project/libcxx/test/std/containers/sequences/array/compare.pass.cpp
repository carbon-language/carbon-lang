//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// bool operator==(array<T, N> const&, array<T, N> const&);   // constexpr in C++20
// bool operator!=(array<T, N> const&, array<T, N> const&);   // constexpr in C++20
// bool operator<(array<T, N> const&, array<T, N> const&);    // constexpr in C++20
// bool operator<=(array<T, N> const&, array<T, N> const&);   // constexpr in C++20
// bool operator>(array<T, N> const&, array<T, N> const&);    // constexpr in C++20
// bool operator>=(array<T, N> const&, array<T, N> const&);   // constexpr in C++20


#include <array>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

TEST_CONSTEXPR_CXX20 bool tests()
{
    {
        typedef std::array<int, 3> C;
        const C c1 = {1, 2, 3};
        const C c2 = {1, 2, 3};
        const C c3 = {3, 2, 1};
        const C c4 = {1, 2, 1};
        assert(testComparisons6(c1, c2, true, false));
        assert(testComparisons6(c1, c3, false, true));
        assert(testComparisons6(c1, c4, false, false));
    }
    {
        typedef std::array<int, 0> C;
        const C c1 = {};
        const C c2 = {};
        assert(testComparisons6(c1, c2, true, false));
    }
    {
        typedef std::array<LessAndEqComp, 3> C;
        const C c1 = {LessAndEqComp(1), LessAndEqComp(2), LessAndEqComp(3)};
        const C c2 = {LessAndEqComp(1), LessAndEqComp(2), LessAndEqComp(3)};
        const C c3 = {LessAndEqComp(3), LessAndEqComp(2), LessAndEqComp(1)};
        const C c4 = {LessAndEqComp(1), LessAndEqComp(2), LessAndEqComp(1)};
        assert(testComparisons6(c1, c2, true, false));
        assert(testComparisons6(c1, c3, false, true));
        assert(testComparisons6(c1, c4, false, false));
    }
    {
        typedef std::array<LessAndEqComp, 0> C;
        const C c1 = {};
        const C c2 = {};
        assert(testComparisons6(c1, c2, true, false));
    }

    return true;
}

int main(int, char**)
{
    tests();
#if TEST_STD_VER >= 20
    static_assert(tests(), "");
#endif
    return 0;
}
