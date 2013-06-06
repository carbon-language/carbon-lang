// RUN: %clang -g -fsanitize=address -O2 %s -o %t
// RUN: ASAN_OPTIONS="sleep_before_dying=1" %t 2>&1 | FileCheck %s

#include <stdlib.h>
int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
  // CHECK: Sleeping for 1 second
}
