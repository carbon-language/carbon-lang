// Test that fake stack (introduced by ASan's use-after-return mode) is included
// in the root set.
// RUN: LSAN_BASE="report_objects=1:use_registers=0"
// RUN: %clangxx_lsan %s -O2 -o %t
// RUN: ASAN_OPTIONS=$ASAN_OPTIONS:detect_stack_use_after_return=1 LSAN_OPTIONS=$LSAN_BASE:"use_stacks=0" not %run %t 2>&1 | FileCheck %s
// RUN: ASAN_OPTIONS=$ASAN_OPTIONS:detect_stack_use_after_return=1 LSAN_OPTIONS=$LSAN_BASE:"use_stacks=1" %run %t 2>&1
// RUN: ASAN_OPTIONS=$ASAN_OPTIONS:detect_stack_use_after_return=1 LSAN_OPTIONS="" %run %t 2>&1

#include <stdio.h>
#include <stdlib.h>

int main() {
  void *stack_var = malloc(1337);
  fprintf(stderr, "Test alloc: %p.\n", stack_var);
  // Take pointer to variable, to ensure it's not optimized into a register.
  fprintf(stderr, "Stack var at: %p.\n", &stack_var);
  // Do not return from main to prevent the pointer from going out of scope.
  exit(0);
}
// CHECK: Test alloc: [[ADDR:.*]].
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: [[ADDR]] (1337 bytes)
// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer:
