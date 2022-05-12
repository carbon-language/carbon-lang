//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <any>

// any() noexcept;

#include <any>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "any_helpers.h"
#include "count_new.h"

int main(int, char**)
{
    {
        static_assert(
            std::is_nothrow_default_constructible<std::any>::value
          , "Must be default constructible"
          );
    }
    {
        struct TestConstexpr : public std::any {
          constexpr TestConstexpr() : std::any() {}
        };
        static TEST_CONSTINIT std::any a;
        (void)a;
    }
    {
        DisableAllocationGuard g; ((void)g);
        const std::any a;
        assertEmpty(a);
    }

  return 0;
}
