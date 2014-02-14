// RUN: %clang_tsan -O1 %s -o %t
// RUN: not %t 2>&1 | FileCheck %s --check-prefix=CHECK-NOZUPP
// RUN: TSAN_OPTIONS="suppressions=%s.supp print_suppressions=1" %t 2>&1 | FileCheck %s --check-prefix=CHECK-SUPP

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <unistd.h>

int *mem;
pthread_mutex_t mtx;

void *Thread1(void *x) {
  pthread_mutex_lock(&mtx);
  free(mem);
  pthread_mutex_unlock(&mtx);
  return NULL;
}

void *Thread2(void *x) {
  sleep(1);
  pthread_mutex_lock(&mtx);
  mem[0] = 42;
  pthread_mutex_unlock(&mtx);
  return NULL;
}

int main() {
  mem = (int*)malloc(100);
  pthread_mutex_init(&mtx, 0);
  pthread_t t;
  pthread_create(&t, NULL, Thread1, NULL);
  Thread2(0);
  pthread_join(t, NULL);
  pthread_mutex_destroy(&mtx);
  return 0;
}

// CHECK-NOZUPP: WARNING: ThreadSanitizer: heap-use-after-free
// CHECK-NOZUPP:   Write of size 4 at {{.*}} by main thread{{.*}}:
// CHECK-NOZUPP:     #0 Thread2
// CHECK-NOZUPP:     #1 main
// CHECK-NOZUPP:   Previous write of size 8 at {{.*}} by thread T1{{.*}}:
// CHECK-NOZUPP:     #0 free
// CHECK-NOZUPP:     #{{(1|2)}} Thread1
// CHECK-NOZUPP: SUMMARY: ThreadSanitizer: heap-use-after-free{{.*}}Thread2
// CHECK-SUPP:   ThreadSanitizer: Matched 1 suppressions
// CHECK-SUPP:    1 race:^Thread2$
