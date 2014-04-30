// Test __msan_set_indirect_call_wrapper.

// RUN: %clangxx_msan -mllvm -msan-wrap-indirect-calls=__msan_wrap_indirect_call \
// RUN:     -mllvm -msan-wrap-indirect-calls-fast=0 \
// RUN:     -O0 -g -rdynamic -Wl,--defsym=__executable_start=0 %s -o %t && %run %t

// This test disables -msan-wrap-indirect-calls-fast, otherwise indirect calls
// inside the same module are short-circuited and are never seen by the wrapper.

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>

extern "C" void __msan_set_indirect_call_wrapper(uintptr_t);

bool done_f, done_g;

void f(void) {
  assert(!done_g);
  done_f = true;
}

void g(void) {
  assert(done_f);
  done_g = true;
}

typedef void (*Fn)(void);
extern "C" Fn my_wrapper(Fn target) {
  if (target == f) return g;
  return target;
}

int main(void) {
  volatile Fn fp;
  fp = &f;
  fp();
  __msan_set_indirect_call_wrapper((uintptr_t)my_wrapper);
  fp();
  return !(done_f && done_g);
}
