// Test that stacks of non-main threads are included in the root set.
// RUN: LSAN_BASE="report_objects=1:use_registers=0"
// RUN: %clangxx_lsan -pthread %s -o %t
// RUN: LSAN_OPTIONS=$LSAN_BASE:"use_stacks=0" %t 2>&1 | FileCheck %s
// RUN: LSAN_OPTIONS=$LSAN_BASE:"use_stacks=1" %t 2>&1
// RUN: LSAN_OPTIONS="" %t 2>&1

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

extern "C"
void *stacks_thread_func(void *arg) {
  int *sync = reinterpret_cast<int *>(arg);
  void *p = malloc(1337);
  fprintf(stderr, "Test alloc: %p.\n", p);
  fflush(stderr);
  __sync_fetch_and_xor(sync, 1);
  while (true)
    pthread_yield();
}

int main() {
  int sync = 0;
  pthread_t thread_id;
  int res = pthread_create(&thread_id, 0, stacks_thread_func, &sync);
  assert(res == 0);
  while (!__sync_fetch_and_xor(&sync, 0))
    pthread_yield();
  return 0;
}
// CHECK: Test alloc: [[ADDR:.*]].
// CHECK: Directly leaked 1337 byte object at [[ADDR]]
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: SUMMARY: LeakSanitizer:
