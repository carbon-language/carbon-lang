// RUN: LSAN_BASE="use_registers=0:use_stacks=0"
// RUN: %clangxx_lsan %s -o %t

// RUN: echo "leak:*LSanTestLeakingFunc*" > %t.supp1
// RUN: LSAN_OPTIONS=$LSAN_BASE:suppressions=%t.supp1 not %t 2>&1 | FileCheck %s

// RUN: echo "leak:%t" > %t.supp2
// RUN: LSAN_OPTIONS=$LSAN_BASE:suppressions="%t.supp2":symbolize=false %t

#include <stdio.h>
#include <stdlib.h>

void LSanTestLeakingFunc() {
  void *p = malloc(666);
  fprintf(stderr, "Test alloc: %p.\n", p);
}

int main() {
  LSanTestLeakingFunc();
  void *q = malloc(1337);
  fprintf(stderr, "Test alloc: %p.\n", q);
  return 0;
}
// CHECK: Suppressions used:
// CHECK: 1 666 *LSanTestLeakingFunc*
// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer: 1337 byte(s) leaked in 1 allocation(s)
