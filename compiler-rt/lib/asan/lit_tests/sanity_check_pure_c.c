// Sanity checking a test in pure C.
// RUN: %clang -g -faddress-sanitizer -O2 %s -o %t
// RUN: %t 2>&1 | FileCheck %s

// Sanity checking a test in pure C with -pie.
// RUN: %clang -g -faddress-sanitizer -O2 %s -pie -o %t
// RUN: %t 2>&1 | FileCheck %s

#include <stdlib.h>
int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
  // CHECK: heap-use-after-free
}
