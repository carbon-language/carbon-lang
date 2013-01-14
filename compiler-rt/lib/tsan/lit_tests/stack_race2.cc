// RUN: %clangxx_tsan -O1 %s -o %t && %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stddef.h>
#include <unistd.h>

void *Thread2(void *a) {
  *(int*)a = 43;
  return 0;
}

void *Thread(void *a) {
  int Var = 42;
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
// CHECK:   Location is stack of thread T1.

