// RUN: %clang_tsan -O1 %s -o %t && not %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <unistd.h>

_Atomic(int*) p;

void *thr(void *a) {
  sleep(1);
  int *pp = __c11_atomic_load(&p, __ATOMIC_RELAXED);
  *pp = 42;
  return 0;
}

int main() {
  pthread_t th;
  pthread_create(&th, 0, thr, p);
  __c11_atomic_store(&p, new int, __ATOMIC_RELAXED);
  pthread_join(th, 0);
}

// CHECK: data race
// CHECK:   Previous write
// CHECK:     #0 operator new
// CHECK:   Location is heap block
// CHECK:     #0 operator new
