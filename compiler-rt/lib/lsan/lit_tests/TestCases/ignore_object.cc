// Test for __lsan_ignore_object().
// RUN: LSAN_BASE="report_objects=1:use_registers=0:use_stacks=0:use_globals=0:use_tls=0:verbosity=2"
// RUN: %clangxx_lsan %s -o %t
// RUN: LSAN_OPTIONS=$LSAN_BASE %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

#include "sanitizer/lsan_interface.h"

int main() {
  {
    // The first malloc call can cause an allocation in libdl. Ignore it here so
    // it doesn't show up in our output.
    __lsan::ScopedDisabler d;
    malloc(1);
  }
  // Explicitly ignored object.
  void **p = new void *;
  // Transitively ignored object.
  *p = malloc(666);
  // Non-ignored object.
  volatile void *q = malloc(1337);
  fprintf(stderr, "Test alloc: %p.\n", p);
  __lsan_ignore_object(p);
  return 0;
}
// CHECK: Test alloc: [[ADDR:.*]].
// CHECK: ignoring heap object at [[ADDR]]
// CHECK: SUMMARY: LeakSanitizer: 1337 byte(s) leaked in 1 allocation(s)
