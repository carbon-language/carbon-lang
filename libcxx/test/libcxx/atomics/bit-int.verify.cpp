//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <atomic>

// Make sure that `std::atomic` doesn't work with `_BitInt`. The intent is to
// disable them for now until their behavior can be designed better later.
// See https://reviews.llvm.org/D84049 for details.

// UNSUPPORTED: apple-clang-12, apple-clang-13, apple-clang-14
// UNSUPPORTED: clang-12, clang-13

// UNSUPPORTED: c++03

#include <atomic>

int main(int, char**)
{
  // expected-error@atomic:*1 {{_Atomic cannot be applied to integer type '_BitInt(32)'}}
  std::atomic<_BitInt(32)> x(42);

  return 0;
}
