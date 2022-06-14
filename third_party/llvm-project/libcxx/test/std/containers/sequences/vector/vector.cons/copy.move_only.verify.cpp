//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that a std::vector containing move-only types can't be copied.

// UNSUPPORTED: c++03 && !stdlib=libc++

#include <vector>

#include "MoveOnly.h"

int main(int, char**)
{
    std::vector<MoveOnly> v;
    std::vector<MoveOnly> copy = v; // expected-error-re@* {{{{(no matching function for call to 'construct_at')|(call to implicitly-deleted copy constructor of 'MoveOnly')|(call to deleted constructor of 'MoveOnly')}}}}
    return 0;
}
