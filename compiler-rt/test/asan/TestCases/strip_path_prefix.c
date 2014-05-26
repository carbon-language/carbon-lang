// RUN: %clang_asan -O2 %s -o %t
// RUN: env ASAN_OPTIONS="strip_path_prefix='/'" not %run %t 2>&1 | FileCheck %s

#include <stdlib.h>
int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
  // Check that paths in error report don't start with slash.
  // CHECK: heap-use-after-free
  // CHECK-NOT: #0 0x{{.*}} ({{[/].*}})
}
