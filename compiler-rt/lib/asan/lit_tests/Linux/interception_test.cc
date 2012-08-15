// ASan interceptor can be accessed with __interceptor_ prefix.

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

extern "C" long __interceptor_strtol(const char *nptr, char **endptr, int base);
extern "C" long strtol(const char *nptr, char **endptr, int base) {
  fprintf(stderr, "my_strtol_interceptor\n");
  return __interceptor_strtol(nptr, endptr, base);
}

int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return (int)strtol(x, 0, 10);
  // CHECK: my_strtol_interceptor
  // CHECK: heap-use-after-free
}
