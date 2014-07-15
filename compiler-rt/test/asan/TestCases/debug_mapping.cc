// Checks that the debugging API returns correct shadow scale and offset.
// RUN: %clangxx_asan -O %s -o %t
// RUN: env ASAN_OPTIONS=verbosity=1 %run %t 2>&1 | FileCheck %s

#include <sanitizer/asan_interface.h>
#include <stdio.h>
#include <stdlib.h>

// printed because of verbosity=1
// CHECK: SHADOW_SCALE: [[SCALE:[0-9]+]]
// CHECK: SHADOW_OFFSET: [[OFFSET:[0-9]+]]

int main() {
  size_t scale, offset;
  __asan_get_shadow_mapping(&scale, &offset);

  fprintf(stderr, "scale: %lx\n", scale);
  fprintf(stderr, "offset: %lx\n", offset);

  // CHECK: scale: [[SCALE]]
  // CHECK: offset: [[OFFSET]]

  return 0;
}
