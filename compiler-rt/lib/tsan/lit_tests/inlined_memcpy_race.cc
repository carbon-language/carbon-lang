// RUN: %clangxx_tsan -O1 %s -o %t && %t 2>&1 | FileCheck %s
// Currently, we don't report a race here:
// http://code.google.com/p/thread-sanitizer/issues/detail?id=16
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int x[4], y[4];

void *Thread1(void *unused) {
  memcpy(x, y, 16);
  return NULL;
}

int main() {
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread1, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  printf("PASS\n");
  return 0;
}

// CHECK-NOT: ThreadSanitizer
