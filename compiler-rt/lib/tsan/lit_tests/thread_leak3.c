// RUN: %clang_tsan -O1 %s -o %t && %t 2>&1 | FileCheck %s
#include <pthread.h>

void *Thread(void *x) {
  return 0;
}

int main() {
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: thread leak
// CHECK: SUMMARY: ThreadSanitizer: thread leak{{.*}}main
