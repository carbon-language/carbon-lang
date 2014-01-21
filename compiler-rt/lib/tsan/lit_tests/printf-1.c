// RUN: %clang_tsan -O2 %s -o %t
// RUN: ASAN_OPTIONS=check_printf=1 %t 2>&1 | FileCheck %s
// RUN: ASAN_OPTIONS=check_printf=0 %t 2>&1 | FileCheck %s
// RUN: %t 2>&1 | FileCheck %s

#include <stdio.h>
int main() {
  volatile char c = '0';
  volatile int x = 12;
  volatile float f = 1.239;
  volatile char s[] = "34";
  printf("%c %d %.3f %s\n", c, x, f, s);
  return 0;
  // Check that printf works fine under Tsan.
  // CHECK: 0 12 1.239 34
}
