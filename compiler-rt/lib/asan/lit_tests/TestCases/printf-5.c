// RUN: %clang_asan -O2 %s -o %t
// RUN: ASAN_OPTIONS=check_printf=1 not %t 2>&1 | FileCheck --check-prefix=CHECK-ON %s
// RUN: ASAN_OPTIONS=check_printf=0 %t 2>&1 | FileCheck --check-prefix=CHECK-OFF %s
// RUN: %t 2>&1 | FileCheck --check-prefix=CHECK-OFF %s

#include <stdio.h>
#include <string.h>
int main() {
  volatile char c = '0';
  volatile int x = 12;
  volatile float f = 1.239;
  volatile char s[] = "34";
  volatile char fmt[2];
  memcpy((char *)fmt, "%c %d %f %s\n", sizeof(fmt));
  printf(fmt, c, x, f, s);
  return 0;
  // Check that format string is sanitized.
  // CHECK-ON: stack-buffer-overflow
  // CHECK-ON-NOT: 0 12 1.239 34
  // CHECK-OFF: 0
}
