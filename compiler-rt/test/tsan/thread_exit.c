// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1
// | FileCheck %s
#include "test.h"

int var;

void *Thread(void *x) {
  fprintf(stderr, "Thread\n");
  pthread_exit(&var);
  return 0;
}

int main() {
  fprintf(stderr, "MAIN\n");
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  void *retval = 0;
  fprintf(stderr, "JOIN\n");
  pthread_join(t, &retval);
  if (retval != &var) {
    fprintf(stderr, "Unexpected return value\n");
    exit(1);
  }
  fprintf(stderr, "PASS\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: PASS
