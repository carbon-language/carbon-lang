//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template <class T, size_t N> T* end(T (&array)[N]);

#include <iterator>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    int ia[] = {1, 2, 3};
    int* i = std::begin(ia);
    int* e = std::end(ia);
    assert(e == ia + 3);
    assert(e - i == 3);

  return 0;
}
