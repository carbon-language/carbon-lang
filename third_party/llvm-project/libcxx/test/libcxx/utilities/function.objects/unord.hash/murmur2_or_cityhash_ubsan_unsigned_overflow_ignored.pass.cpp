//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that UBSAN doesn't generate unsigned integer overflow diagnostics
// from within the hashing internals.

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <utility>

#include "test_macros.h"

typedef std::__murmur2_or_cityhash<uint32_t> Hash32;
typedef std::__murmur2_or_cityhash<uint64_t> Hash64;

void test(const void* key, int len) {
  for (int i=1; i <= len; ++i) {
    Hash32 h1;
    Hash64 h2;
    DoNotOptimize(h1(key, i));
    DoNotOptimize(h2(key, i));
  }
}

int main(int, char**) {
  const std::string TestCases[] = {
      "abcdaoeuaoeclaoeoaeuaoeuaousaotehu]+}sthoasuthaoesutahoesutaohesutaoeusaoetuhasoetuhaoseutaoseuthaoesutaohes",
      "00000000000000000000000000000000000000000000000000000000000000000000000",
      "1237546895+54+4554985416849484213464984765465464654564565645645646546456546546"
  };
  const size_t NumCases = sizeof(TestCases)/sizeof(TestCases[0]);
  for (size_t i=0; i < NumCases; ++i)
    test(TestCases[i].data(), TestCases[i].length());

  return 0;
}
