// UNSUPPORTED: armv6m-target-arch
// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_bswapdi2

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern uint64_t __bswapdi2(uint64_t);

int test__bswapdi2(uint64_t a, uint64_t expected) {
  uint64_t actual = __bswapdi2(a);
  if (actual != expected)
    printf("error in test__bswapsi2(0x%0llX) = 0x%0llX, expected 0x%0llX\n", a,
           actual, expected);
  return actual != expected;
}

int main() {
  if (test__bswapdi2(0x123456789ABCDEF0LL, 0xF0DEBC9A78563412LL))
    return 1;
  if (test__bswapdi2(0x0000000100000002LL, 0x0200000001000000LL))
    return 1;
  return 0;
}
