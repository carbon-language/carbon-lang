// RUN: %clangxx_asan -O0 -fsanitize=use-after-scope %s -o %t && %t

#include <stdio.h>

int main() {
  int *p = 0;
  // Variable goes in and out of scope.
  for (int i = 0; i < 3; i++) {
    int x = 0;
    p = &x;
  }
  printf("PASSED\n");
  return 0;
}
