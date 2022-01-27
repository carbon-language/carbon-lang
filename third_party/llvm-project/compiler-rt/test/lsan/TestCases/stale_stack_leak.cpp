// Test that out-of-scope local variables are ignored by LSan.
// RUN: LSAN_BASE="report_objects=1:use_registers=0:use_stacks=1"

// LSan-in-ASan fails at -O0 on aarch64, because the stack use-after-return
// instrumentation stashes the argument to `PutPointerOnStaleStack` on the stack
// in order to conditionally call __asan_stack_malloc. This subverts our
// expectations for this test, where we assume the pointer is never stashed
// except at the bottom of the dead frame. Building at -O1 or greater solves
// this problem, because the compiler is smart enough to stash the argument in a
// callee-saved register for rematerialization instead.
// RUN: %clangxx_lsan -O1 %s -o %t

// RUN: %env_lsan_opts=$LSAN_BASE not %run %t 2>&1 | FileCheck %s
// RUN: %env_lsan_opts=$LSAN_BASE":exitcode=0" %run %t 2>&1 | FileCheck --check-prefix=CHECK-sanity %s
//
// x86 passes parameters through stack that may lead to false negatives
// The same applies to s390x register save areas.
// UNSUPPORTED: x86,i686,powerpc64,arm,s390x

#include <stdio.h>
#include <stdlib.h>
#include "sanitizer_common/print_address.h"

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
  print_address("Test alloc: ", 1, locals[0]);
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
__attribute__((no_sanitize_address))
void ConfirmPointerHasSurvived() {
  print_address("Value after LSan: ", 1, *pp);
}
// CHECK: Test alloc: [[ADDR:0x[0-9,a-f]+]]
// CHECK-sanity: Test alloc: [[ADDR:0x[0-9,a-f]+]]
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: [[ADDR]] (1337 bytes)
// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer:
// CHECK-sanity: Value after LSan: [[ADDR]]
