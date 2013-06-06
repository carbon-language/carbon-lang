// If user provides his own libc functions, ASan doesn't
// intercept these functions.

// RUN: %clangxx_asan -m64 -O0 %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -m64 -O1 %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -m64 -O2 %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -m64 -O3 %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -m32 -O0 %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -m32 -O1 %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -m32 -O2 %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -m32 -O3 %s -o %t && %t 2>&1 | FileCheck %s
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
