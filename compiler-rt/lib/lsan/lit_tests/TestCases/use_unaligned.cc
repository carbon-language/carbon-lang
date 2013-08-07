// Test that unaligned pointers are detected correctly.
// RUN: LSAN_BASE="report_objects=1:use_stacks=0:use_registers=0"
// RUN: %clangxx_lsan %s -o %t
// RUN: LSAN_OPTIONS=$LSAN_BASE:"use_unaligned=0" not %t 2>&1 | FileCheck %s
// RUN: LSAN_OPTIONS=$LSAN_BASE:"use_unaligned=1" %t 2>&1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *arr[2];

int main() {
  void *p = malloc(1337);
  fprintf(stderr, "Test alloc: %p.\n", p);
  char *char_arr = (char *)arr;
  memcpy(char_arr + 1, &p, sizeof(p));
  return 0;
}
// CHECK: Test alloc: [[ADDR:.*]].
// CHECK: Directly leaked 1337 byte object at [[ADDR]]
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: SUMMARY: LeakSanitizer:
