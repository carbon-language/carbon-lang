// RUN: %clang_hwasan  %s -o %t && not %run %t 2>&1 | FileCheck %s

// REQUIRES: stable-runtime
// TODO: test more cases.

#include <stdlib.h>
#include <stdio.h>
#include <sanitizer/hwasan_interface.h>

int main() {
  __hwasan_enable_allocator_tagging();
  char * volatile x = (char*)malloc(30);
  x[40] = 42;
// CHECK: is located 10 bytes to the right of 30-byte region
  free(x);
}
