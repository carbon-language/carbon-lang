// RUN: %clangxx_tsan -O1 %s -o %t && %t 2>&1 | FileCheck %s
#include "java.h"
#include <unistd.h>

jptr varaddr;
jptr lockaddr;

void *Thread(void *p) {
  __tsan_java_mutex_lock(lockaddr);
  __tsan_java_mutex_lock(lockaddr);
  *(int*)varaddr = 42;
  int rec = __tsan_java_mutex_unlock_rec(lockaddr);
  if (rec != 2) {
    printf("FAILED 0 rec=%d\n", rec);
    exit(1);
  }
  sleep(2);
  __tsan_java_mutex_lock_rec(lockaddr, rec);
  if (*(int*)varaddr != 43) {
    printf("FAILED 3 var=%d\n", *(int*)varaddr);
    exit(1);
  }
  __tsan_java_mutex_unlock(lockaddr);
  __tsan_java_mutex_unlock(lockaddr);
  return 0;
}

int main() {
  int const kHeapSize = 1024 * 1024;
  void *jheap = malloc(kHeapSize);
  __tsan_java_init((jptr)jheap, kHeapSize);
  const int kBlockSize = 16;
  __tsan_java_alloc((jptr)jheap, kBlockSize);
  varaddr = (jptr)jheap;
  *(int*)varaddr = 0;
  lockaddr = (jptr)jheap + 8;
  pthread_t th;
  pthread_create(&th, 0, Thread, 0);
  sleep(1);
  __tsan_java_mutex_lock(lockaddr);
  if (*(int*)varaddr != 42) {
    printf("FAILED 1 var=%d\n", *(int*)varaddr);
    exit(1);
  }
  *(int*)varaddr = 43;
  __tsan_java_mutex_unlock(lockaddr);
  pthread_join(th, 0);
  __tsan_java_free((jptr)jheap, kBlockSize);
  printf("OK\n");
  return __tsan_java_fini();
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK-NOT: FAILED
