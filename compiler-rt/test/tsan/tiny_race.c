// RUN: %clang_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include <pthread.h>
#include <unistd.h>

int Global;

void *Thread1(void *x) {
  sleep(1);
  Global = 42;
  return x;
}

int main() {
  pthread_t t;
  pthread_create(&t, 0, Thread1, 0);
  Global = 43;
  pthread_join(t, 0);
  return Global;
}

// CHECK: WARNING: ThreadSanitizer: data race
