// RUN: %clangxx_asan -m64 -O0 %s -fsanitize-address-zero-base-shadow -pie -o %t && %t 2>&1 | %symbolize | FileCheck %s

// Test the time() interceptor. Also includes a regression test for time(NULL),
// which caused ASan to crash in the zero-based shadow mode.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  time_t *tm = (time_t*)malloc(sizeof(time_t));
  free(tm);
  time_t t = time(NULL);
  fprintf(stderr, "Time: %s\n", ctime(&t));
  // CHECK: {{Time: .* .* .*}}
  t = time(tm);
  printf("Time: %s\n", ctime(&t));
  // CHECK: use-after-free
  return 0;
}
