// Test that preloaded runtime works with unsanitized executables.
//
// RUN: %clangxx %s -o %t
// RUN: LD_PRELOAD=%shared_libasan not %run %t 2>&1 | FileCheck %s

// REQUIRES: asan-dynamic-runtime

#include <stdlib.h>

extern "C" void *memset(void *p, int val, size_t n);

void do_access(void *p) {
  // CHECK: AddressSanitizer: heap-buffer-overflow
  memset(p, 0, 2);
}

int main(int argc, char **argv) {
  void *p = malloc(1);
  do_access(p);
  return 0;
}
