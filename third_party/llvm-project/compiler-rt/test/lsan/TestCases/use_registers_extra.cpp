// Test that registers of running threads are included in the root set.
// RUN: %clangxx_lsan -pthread %s -o %t
// RUN: %env_lsan_opts="report_objects=1:use_stacks=0:use_registers=0" not %run %t 2>&1 | FileCheck %s
// RUN: %env_lsan_opts="report_objects=1:use_stacks=0:use_registers=1" %run %t 2>&1
// RUN: %env_lsan_opts="" %run %t 2>&1

// FIXME: Support more platforms.
// REQUIRES: x86-target-arch && linux

#include "sanitizer_common/print_address.h"
#include <assert.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void *registers_thread_func(void *arg) {
  int *sync = reinterpret_cast<int *>(arg);
  void *p = malloc(1337);
  print_address("Test alloc: ", 1, p);
  fflush(stderr);

  // To store the pointer, choose a register which is unlikely to be reused by
  // a function call.
#if defined(__i386__)
  asm(R"(
    movd %0, %%xmm0
    mov $0, %0
  )"
      :
      : "r"(p));
#elif defined(__x86_64__)
  asm(R"(
    movq %0, %%xmm0
    mov $0, %0
  )"
      :
      : "r"(p));
#else
#error "Test is not supported on this architecture."
#endif

  __sync_fetch_and_xor(sync, 1);
  while (true)
    sched_yield();
}

int main() {
  int sync = 0;
  pthread_t thread_id;
  int res = pthread_create(&thread_id, 0, registers_thread_func, &sync);
  assert(res == 0);
  while (!__sync_fetch_and_xor(&sync, 0))
    sched_yield();
  return 0;
}
// CHECK: Test alloc: [[ADDR:0x[0-9,a-f]+]]
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: [[ADDR]] (1337 bytes)
// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer:
