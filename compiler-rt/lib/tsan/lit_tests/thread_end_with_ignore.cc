// RUN: %clangxx_tsan -O1 %s -o %t && %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>

extern "C" void AnnotateIgnoreReadsBegin(const char *f, int l);

void *Thread(void *x) {
  AnnotateIgnoreReadsBegin("", 0);
  return 0;
}

int main() {
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  pthread_join(t, 0);
}

// CHECK: ThreadSanitizer: thread T1 finished with ignores enabled

