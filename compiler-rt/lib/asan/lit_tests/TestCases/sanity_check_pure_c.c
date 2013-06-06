// Sanity checking a test in pure C.
// RUN: %clang -g -fsanitize=address -O2 %s -o %t
// RUN: %t 2>&1 | %symbolize | FileCheck %s

// Sanity checking a test in pure C with -pie.
// RUN: %clang -g -fsanitize=address -O2 %s -pie -o %t
// RUN: %t 2>&1 | %symbolize | FileCheck %s

#include <stdlib.h>
int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
  // CHECK: heap-use-after-free
  // CHECK: free
  // CHECK: main{{.*}}sanity_check_pure_c.c:[[@LINE-4]]
  // CHECK: malloc
  // CHECK: main{{.*}}sanity_check_pure_c.c:[[@LINE-7]]
}
