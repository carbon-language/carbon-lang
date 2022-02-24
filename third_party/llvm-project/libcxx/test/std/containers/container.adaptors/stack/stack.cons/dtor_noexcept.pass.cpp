//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <stack>

// ~stack() // implied noexcept;

// UNSUPPORTED: c++03

#include <stack>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"

int main(int, char**)
{
    {
        typedef std::stack<MoveOnly> C;
        static_assert(std::is_nothrow_destructible<C>::value, "");
    }

  return 0;
}
