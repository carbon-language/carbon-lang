// Test that LargeMmapAllocator's chunks aren't reachable via some internal data structure.
// RUN: LSAN_BASE="report_objects=1:use_stacks=0:use_registers=0"
// RUN: %clangxx_lsan %s -o %t
// RUN: LSAN_OPTIONS=$LSAN_BASE not %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

int main() {
  // maxsize in primary allocator is always less than this (1 << 25).
  void *large_alloc = malloc(33554432);
  fprintf(stderr, "Test alloc: %p.\n", large_alloc);
  return 0;
}
// CHECK: Test alloc: [[ADDR:.*]].
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: [[ADDR]] (33554432 bytes)
// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer:
