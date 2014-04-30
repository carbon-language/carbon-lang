// RUN: %clang_tsan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <unistd.h>

void *Thread(void *x) {
  return 0;
}

int main() {
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  sleep(1);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: thread leak
// CHECK: SUMMARY: ThreadSanitizer: thread leak{{.*}}main
