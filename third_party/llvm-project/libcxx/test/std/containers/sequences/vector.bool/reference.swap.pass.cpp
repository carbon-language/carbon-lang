//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// static void swap(reference x, reference y) noexcept;

#include <vector>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{

    bool a[] = {false, true, false, true};
    bool* an = a + sizeof(a)/sizeof(a[0]);

    std::vector<bool> v(a, an);
    std::vector<bool>::reference r1 = v[0];
    std::vector<bool>::reference r2 = v[3];

#if TEST_STD_VER >= 11
    static_assert((noexcept(v.swap(r1,r2))), "");
#endif

    assert(!r1);
    assert( r2);
    v.swap(r1, r2);
    assert( r1);
    assert(!r2);

  return 0;
}
