// Test that uninitialized globals are included in the root set.
// RUN: LSAN_BASE="report_objects=1:use_stacks=0:use_registers=0"
// RUN: %clangxx_lsan %s -o %t
// RUN: LSAN_OPTIONS=$LSAN_BASE:"use_globals=0" not %t 2>&1 | FileCheck %s
// RUN: LSAN_OPTIONS=$LSAN_BASE:"use_globals=1" %t 2>&1
// RUN: LSAN_OPTIONS="" %t 2>&1

#include <stdio.h>
#include <stdlib.h>

void *bss_var;

int main() {
  bss_var = malloc(1337);
  fprintf(stderr, "Test alloc: %p.\n", bss_var);
  return 0;
}
// CHECK: Test alloc: [[ADDR:.*]].
// CHECK: Directly leaked 1337 byte object at [[ADDR]]
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: SUMMARY: LeakSanitizer:
