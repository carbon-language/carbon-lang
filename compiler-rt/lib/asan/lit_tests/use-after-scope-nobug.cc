// RUN: %clangxx_asan -m64 -O0 -fsanitize=use-after-scope %s -o %t && \
// RUN:     %t 2>&1 | %symbolize | FileCheck %s

#include <stdio.h>

int main() {
  int *p = 0;
  // Variable goes in and out of scope.
  for (int i = 0; i < 3; i++) {
    int x = 0;
    p = &x;
  }
  printf("PASSED\n");
  // CHECK: PASSED
  return 0;
}
