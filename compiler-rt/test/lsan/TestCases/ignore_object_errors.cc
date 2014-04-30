// Test for incorrect use of __lsan_ignore_object().
// RUN: LSAN_BASE="verbosity=2"
// RUN: %clangxx_lsan %s -o %t
// RUN: LSAN_OPTIONS=$LSAN_BASE ASAN_OPTIONS=$ASAN_OPTIONS:verbosity=2 %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

#include "sanitizer/lsan_interface.h"

int main() {
  void *p = malloc(1337);
  fprintf(stderr, "Test alloc: %p.\n", p);
  __lsan_ignore_object(p);
  __lsan_ignore_object(p);
  free(p);
  __lsan_ignore_object(p);
  return 0;
}
// CHECK: Test alloc: [[ADDR:.*]].
// CHECK: heap object at [[ADDR]] is already being ignored
// CHECK: no heap object found at [[ADDR]]
