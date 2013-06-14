// Test for ScopedDisabler.
// RUN: LSAN_BASE="report_objects=1:use_registers=0:use_stacks=0:use_globals=0:use_tls=0"
// RUN: %clangxx_lsan %s -o %t
// RUN: LSAN_OPTIONS=$LSAN_BASE %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

#include "sanitizer/lsan_interface.h"

int main() {
  void **p;
  {
    __lsan::ScopedDisabler d;
    p = new void *;
  }
  *reinterpret_cast<void **>(p) = malloc(666);
  void *q = malloc(1337);
  // Break optimization.
  fprintf(stderr, "Test alloc: %p.\n", q);
  return 0;
}
// CHECK: SUMMARY: LeakSanitizer: 1337 byte(s) leaked in 1 allocation(s)
