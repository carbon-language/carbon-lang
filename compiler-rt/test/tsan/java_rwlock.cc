// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "java.h"
#include <unistd.h>

jptr varaddr;
jptr lockaddr;

void *Thread(void *p) {
  sleep(1);
  __tsan_java_mutex_read_lock(lockaddr);
  *(int*)varaddr = 42;
  __tsan_java_mutex_read_unlock(lockaddr);
  return 0;
}

int main() {
  int const kHeapSize = 1024 * 1024;
  jptr jheap = (jptr)malloc(kHeapSize + 8) + 8;
  __tsan_java_init(jheap, kHeapSize);
  const int kBlockSize = 16;
  __tsan_java_alloc(jheap, kBlockSize);
  varaddr = jheap;
  lockaddr = jheap + 8;
  pthread_t th;
  pthread_create(&th, 0, Thread, 0);
  __tsan_java_mutex_lock(lockaddr);
  *(int*)varaddr = 43;
  __tsan_java_mutex_unlock(lockaddr);
  pthread_join(th, 0);
  __tsan_java_free(jheap, kBlockSize);
  printf("DONE\n");
  return __tsan_java_fini();
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: DONE
