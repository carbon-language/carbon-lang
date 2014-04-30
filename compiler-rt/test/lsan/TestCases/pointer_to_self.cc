// Regression test: pointers to self should not confuse LSan into thinking the
// object is indirectly leaked. Only external pointers count.
// RUN: LSAN_BASE="report_objects=1:use_registers=0"
// RUN: %clangxx_lsan %s -o %t
// RUN: LSAN_OPTIONS=$LSAN_BASE:"use_stacks=0" not %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

int main() {
  void *p = malloc(1337);
  *reinterpret_cast<void **>(p) = p;
  fprintf(stderr, "Test alloc: %p.\n", p);
}
// CHECK: Test alloc: [[ADDR:.*]].
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: [[ADDR]] (1337 bytes)
// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer:
