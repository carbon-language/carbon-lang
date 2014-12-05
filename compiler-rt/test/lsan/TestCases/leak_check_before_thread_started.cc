// Regression test for http://llvm.org/bugs/show_bug.cgi?id=21621
// This test relies on timing between threads, so any failures will be flaky.
// RUN: LSAN_BASE="use_stacks=0:use_registers=0"
// RUN: %clangxx_lsan %s -o %t
// RUN: %run %t
#include <assert.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

void *func(void *arg) {
  sleep(1);
  free(arg);
  return 0;
}

void create_detached_thread() {
  pthread_t thread_id;
  pthread_attr_t attr;

  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

  void *arg = malloc(1337);
  assert(arg);
  int res = pthread_create(&thread_id, &attr, func, arg);
  assert(res == 0);
}

int main() {
  create_detached_thread();
}
