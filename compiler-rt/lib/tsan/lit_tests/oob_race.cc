// RUN: %clangxx_tsan -O1 %s -o %t && not %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>

const long kOffset = 64*1024;

void *Thread(void *p) {
  ((char*)p)[-kOffset] = 43;
  return 0;
}

int main() {
  char *volatile p0 = new char[16];
  delete[] p0;
  char *p = new char[32];
  pthread_t th;
  pthread_create(&th, 0, Thread, p);
  p[-kOffset] = 42;
  pthread_join(th, 0);
}

// Used to crash with CHECK failed.
// CHECK: WARNING: ThreadSanitizer: data race

