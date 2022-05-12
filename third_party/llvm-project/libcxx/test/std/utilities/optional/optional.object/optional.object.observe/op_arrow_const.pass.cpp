//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// constexpr const T* optional<T>::operator->() const;

#include <optional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using std::optional;

struct X
{
    constexpr int test() const {return 3;}
};

struct Y
{
    int test() const noexcept {return 2;}
};

struct Z
{
    const Z* operator&() const;
    constexpr int test() const {return 1;}
};

int main(int, char**)
{
    {
        const std::optional<X> opt; ((void)opt);
        ASSERT_SAME_TYPE(decltype(opt.operator->()), X const*);
        // ASSERT_NOT_NOEXCEPT(opt.operator->());
        // FIXME: This assertion fails with GCC because it can see that
        // (A) operator->() is constexpr, and
        // (B) there is no path through the function that throws.
        // It's arguable if this is the correct behavior for the noexcept
        // operator.
        // Regardless this function should still be noexcept(false) because
        // it has a narrow contract.
    }
    {
        constexpr optional<X> opt(X{});
        static_assert(opt->test() == 3, "");
    }
    {
        constexpr optional<Y> opt(Y{});
        assert(opt->test() == 2);
    }
    {
        constexpr optional<Z> opt(Z{});
        static_assert(opt->test() == 1, "");
    }

    return 0;
}
