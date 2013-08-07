// Test that stack of main thread is included in the root set.
// RUN: LSAN_BASE="report_objects=1:use_registers=0"
// RUN: %clangxx_lsan %s -o %t
// RUN: LSAN_OPTIONS=$LSAN_BASE:"use_stacks=0" not %t 2>&1 | FileCheck %s
// RUN: LSAN_OPTIONS=$LSAN_BASE:"use_stacks=1" %t 2>&1
// RUN: LSAN_OPTIONS="" %t 2>&1

#include <stdio.h>
#include <stdlib.h>

int main() {
  void *stack_var = malloc(1337);
  fprintf(stderr, "Test alloc: %p.\n", stack_var);
  // Do not return from main to prevent the pointer from going out of scope.
  exit(0);
}
// CHECK: Test alloc: [[ADDR:.*]].
// CHECK: Directly leaked 1337 byte object at [[ADDR]]
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: SUMMARY: LeakSanitizer:
