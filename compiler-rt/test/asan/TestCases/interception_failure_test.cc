// If user provides his own libc functions, ASan doesn't
// intercept these functions.

// RUN: %clangxx_asan -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && %run %t 2>&1 | FileCheck %s
// On Windows, defining strtoll results in linker errors.
// XFAIL: freebsd,win32
#include <stdlib.h>
#include <stdio.h>

extern "C" long strtol(const char *nptr, char **endptr, int base) {
  fprintf(stderr, "my_strtol_interceptor\n");
  return 0;
}

int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return (int)strtol(x, 0, 10);
  // CHECK: my_strtol_interceptor
  // CHECK-NOT: heap-use-after-free
}
