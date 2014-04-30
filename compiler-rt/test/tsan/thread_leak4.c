// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>

void *Thread(void *x) {
  sleep(10);
  return 0;
}

int main() {
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  printf("OK\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer: thread leak
