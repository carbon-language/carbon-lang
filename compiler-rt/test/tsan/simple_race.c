// RUN: %clang_tsan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

int Global;

void *Thread1(void *x) {
  sleep(1);
  Global = 42;
  return NULL;
}

void *Thread2(void *x) {
  Global = 43;
  return NULL;
}

int main() {
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
