// RUN: %clangxx_tsan -O1 %s -o %t && not %t 2>&1 | FileCheck %s
#include "java.h"

jptr varaddr;
jptr varaddr2;

void *Thread(void *p) {
  sleep(1);
  *(int*)varaddr2 = 42;
  return 0;
}

int main() {
  int const kHeapSize = 1024 * 1024;
  void *jheap = malloc(kHeapSize);
  __tsan_java_init((jptr)jheap, kHeapSize);
  const int kBlockSize = 64;
  int const kMove = 1024;
  __tsan_java_alloc((jptr)jheap, kBlockSize);
  varaddr = (jptr)jheap + 16;
  varaddr2 = varaddr + kMove;
  pthread_t th;
  pthread_create(&th, 0, Thread, 0);
  *(int*)varaddr = 43;
  __tsan_java_move(varaddr, varaddr2, kBlockSize);
  pthread_join(th, 0);
  __tsan_java_free(varaddr2, kBlockSize);
  return __tsan_java_fini();
}

// CHECK: WARNING: ThreadSanitizer: data race
