// RUN: %clang_asan -O2 %s -o %t
// We need replace_str=0 and replace_intrin=0 to avoid reporting errors in
// strlen() and memcpy() called by puts().
// RUN: %env_asan_opts=replace_str=0:replace_intrin=0:check_printf=1 not %run %t 2>&1 | FileCheck --check-prefix=CHECK-ON %s
// RUN: %env_asan_opts=replace_str=0:replace_intrin=0 not %run %t 2>&1 | FileCheck --check-prefix=CHECK-ON %s

// FIXME: printf is not intercepted on Windows yet.
// XFAIL: win32

#include <stdio.h>
int main() {
  volatile char c = '0';
  volatile int x = 12;
  volatile float f = 1.239;
  volatile char s[] = "34";
  volatile char buf[2];
  sprintf((char *)buf, "%c %d %.3f %s\n", c, x, f, s);
  puts((const char *)buf);
  return 0;
  // Check that size of output buffer is sanitized.
  // CHECK-ON: stack-buffer-overflow
  // CHECK-ON-NOT: 0 12 1.239 34
}
