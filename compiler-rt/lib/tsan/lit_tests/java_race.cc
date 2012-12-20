// RUN: %clangxx_tsan -O1 %s -o %t && %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdlib.h>

extern "C" {
typedef unsigned long jptr;  // NOLINT
void __tsan_java_init(jptr heap_begin, jptr heap_size);
int  __tsan_java_fini();
void __tsan_java_alloc(jptr ptr, jptr size);
void __tsan_java_free(jptr ptr, jptr size);
void __tsan_java_move(jptr src, jptr dst, jptr size);
void __tsan_java_mutex_lock(jptr addr);
void __tsan_java_mutex_unlock(jptr addr);
void __tsan_java_mutex_read_lock(jptr addr);
void __tsan_java_mutex_read_unlock(jptr addr);
}

void *Thread(void *p) {
  *(int*)p = 42;
  return 0;
}

int main() {
  int const kHeapSize = 1024 * 1024;
  void *jheap = malloc(kHeapSize);
  __tsan_java_init((jptr)jheap, kHeapSize);
  const int kBlockSize = 16;
  __tsan_java_alloc((jptr)jheap, kBlockSize);
  pthread_t th;
  pthread_create(&th, 0, Thread, jheap);
  *(int*)jheap = 43;
  pthread_join(th, 0);
  __tsan_java_free((jptr)jheap, kBlockSize);
  return __tsan_java_fini();
}

// CHECK: WARNING: ThreadSanitizer: data race
