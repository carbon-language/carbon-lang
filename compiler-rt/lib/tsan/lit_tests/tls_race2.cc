// RUN: %clangxx_tsan -O1 %s -o %t && not %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stddef.h>
#include <unistd.h>

void *Thread2(void *a) {
  sleep(1);
  *(int*)a = 43;
  return 0;
}

void *Thread(void *a) {
  static __thread int Var = 42;
  pthread_t t;
  pthread_create(&t, 0, Thread2, &Var);
  Var = 42;
  pthread_join(t, 0);
  return 0;
}

int main() {
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  pthread_join(t, 0);
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Location is TLS of thread T1.

