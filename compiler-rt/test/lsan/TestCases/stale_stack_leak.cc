// Test that out-of-scope local variables are ignored by LSan.
// RUN: LSAN_BASE="report_objects=1:use_registers=0:use_stacks=1"
// RUN: %clangxx_lsan %s -o %t
// RUN: LSAN_OPTIONS=$LSAN_BASE not %run %t 2>&1 | FileCheck %s
// RUN: LSAN_OPTIONS=$LSAN_BASE":exitcode=0" %run %t 2>&1 | FileCheck --check-prefix=CHECK-sanity %s

#include <stdio.h>
#include <stdlib.h>

void **pp;

// Put pointer far enough on the stack that LSan has space to run in without
// overwriting it.
// Hopefully the argument p will be passed on a register, saving us from false
// negatives.
__attribute__((noinline))
void *PutPointerOnStaleStack(void *p) {
  void *locals[2048];
  locals[0] = p;
  pp = &locals[0];
  fprintf(stderr, "Test alloc: %p.\n", locals[0]);
  return 0;
}

int main() {
  PutPointerOnStaleStack(malloc(1337));
  return 0;
}

// This must run after LSan, to ensure LSan didn't overwrite the pointer before
// it had a chance to see it. If LSan is invoked with atexit(), this works.
// Otherwise, we need a different method.
__attribute__((destructor))
void ConfirmPointerHasSurvived() {
  fprintf(stderr, "Value after LSan: %p.\n", *pp);
}
// CHECK: Test alloc: [[ADDR:.*]].
// CHECK-sanity: Test alloc: [[ADDR:.*]].
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: [[ADDR]] (1337 bytes)
// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer:
// CHECK-sanity: Value after LSan: [[ADDR]].
