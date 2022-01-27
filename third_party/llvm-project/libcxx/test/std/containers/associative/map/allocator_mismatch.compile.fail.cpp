//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>
//   The container's value type must be the same as the allocator's value type

#include <map>

int main(int, char**)
{
    std::map<int, int, std::less<int>, std::allocator<long> > m;

  return 0;
}
