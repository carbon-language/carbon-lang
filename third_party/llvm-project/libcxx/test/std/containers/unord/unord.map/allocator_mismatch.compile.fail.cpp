//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>
//   The container's value type must be the same as the allocator's value type

#include <unordered_map>

int main(int, char**)
{
    std::unordered_map<int, int, std::hash<int>, std::less<int>, std::allocator<long> > m;

  return 0;
}
