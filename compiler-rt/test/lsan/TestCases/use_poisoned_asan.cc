// ASan-poisoned memory should be ignored if use_poisoned is false.
// REQUIRES: asan
// RUN: LSAN_BASE="report_objects=1:use_stacks=0:use_registers=0"
// RUN: %clangxx_lsan %s -o %t
// RUN: LSAN_OPTIONS=$LSAN_BASE:"use_poisoned=0" not %t 2>&1 | FileCheck %s
// RUN: LSAN_OPTIONS=$LSAN_BASE:"use_poisoned=1" %t 2>&1

#include <stdio.h>
#include <stdlib.h>
#include <sanitizer/asan_interface.h>
#include <assert.h>

void **p;

int main() {
  p = new void *;
  *p = malloc(1337);
  fprintf(stderr, "Test alloc: %p.\n", *p);
  __asan_poison_memory_region(p, sizeof(*p));
  return 0;
}
// CHECK: Test alloc: [[ADDR:.*]].
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: [[ADDR]] (1337 bytes)
// CHECK: SUMMARY: AddressSanitizer:
