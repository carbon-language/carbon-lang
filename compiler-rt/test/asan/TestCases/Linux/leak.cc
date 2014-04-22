// Minimal test for LeakSanitizer+AddressSanitizer.
// REQUIRES: asan-64-bits
//
// RUN: %clangxx_asan  %s -o %t
// RUN: ASAN_OPTIONS=detect_leaks=1 not %t  2>&1 | FileCheck %s
// RUN:                             not %t  2>&1 | FileCheck %s
// RUN: ASAN_OPTIONS=detect_leaks=0     %t
#include <stdio.h>
int *t;

int main(int argc, char **argv) {
  t = new int[argc - 1];
  printf("t: %p\n", t);
  t = 0;
}
// CHECK: LeakSanitizer: detected memory leaks
