// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

void *Thread(void *x) {
  return 0;
}

int main() {
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  sleep(1);
  pthread_detach(t);
  printf("PASS\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer: thread leak
// CHECK: PASS
