// RUN: %clang_tsan -O1 %s -o %t && TSAN_OPTIONS="$TSAN_OPTIONS halt_on_error=1" %deflake %run %t | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

int X;

void *Thread(void *x) {
  sleep(1);
  X = 42;
  return 0;
}

int main() {
  fprintf(stderr, "BEFORE\n");
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  X = 43;
  pthread_join(t, 0);
  fprintf(stderr, "AFTER\n");
  return 0;
}

// CHECK: BEFORE
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK-NOT: AFTER

