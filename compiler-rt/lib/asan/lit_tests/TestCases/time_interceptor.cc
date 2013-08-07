// RUN: %clangxx_asan -O0 %s -o %t && not %t 2>&1 | FileCheck %s

// Test the time() interceptor.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  time_t *tm = (time_t*)malloc(sizeof(time_t));
  free(tm);
  time_t t = time(tm);
  printf("Time: %s\n", ctime(&t));  // NOLINT
  // CHECK: use-after-free
  return 0;
}
