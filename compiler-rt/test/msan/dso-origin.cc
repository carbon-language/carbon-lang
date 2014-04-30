// Build a library with origin tracking and an executable w/o origin tracking.
// Test that origin tracking is enabled at runtime.
// RUN: %clangxx_msan -fsanitize-memory-track-origins -m64 -O0 %p/SharedLibs/dso-origin-so.cc \
// RUN:     -fPIC -shared -o %t-so.so
// RUN: %clangxx_msan -m64 -O0 %s %t-so.so -o %t && not %run %t 2>&1 | FileCheck %s

#include <stdlib.h>

#include "SharedLibs/dso-origin.h"

int main(int argc, char **argv) {
  int *x = (int *)my_alloc(sizeof(int));
  my_access(x);
  delete x;

  // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // CHECK: {{#0 0x.* in my_access .*dso-origin-so.cc:}}
  // CHECK: {{#1 0x.* in main .*dso-origin.cc:}}[[@LINE-5]]
  // CHECK: Uninitialized value was created by a heap allocation
  // CHECK: {{#0 0x.* in .*malloc}}
  // CHECK: {{#1 0x.* in my_alloc .*dso-origin-so.cc:}}
  // CHECK: {{#2 0x.* in main .*dso-origin.cc:}}[[@LINE-10]]
  // CHECK: SUMMARY: MemorySanitizer: use-of-uninitialized-value {{.*dso-origin-so.cc:.* my_access}}
  return 0;
}
