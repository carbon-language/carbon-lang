//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <queue>

// priority_queue()
//        noexcept(is_nothrow_default_constructible<container_type>::value &&
//                 is_nothrow_default_constructible<Compare>::value);

// This tests a conforming extension


#include <queue>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"

int main(int, char**)
{
#if defined(_LIBCPP_VERSION)
    {
        typedef std::priority_queue<MoveOnly> C;
        static_assert(std::is_nothrow_default_constructible<C>::value, "");
    }
#endif

  return 0;
}
