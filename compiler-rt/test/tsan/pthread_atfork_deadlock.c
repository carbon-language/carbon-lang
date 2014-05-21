// RUN: %clang_tsan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// Regression test for
// https://code.google.com/p/thread-sanitizer/issues/detail?id=61
// When the data race was reported, pthread_atfork() handler used to be
// executed which caused another race report in the same thread, which resulted
// in a deadlock.
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

int glob = 0;

void *worker(void *unused) {
  sleep(1);
  glob++;
  return NULL;
}

void atfork() {
  fprintf(stderr, "ATFORK\n");
  glob++;
}

int main() {
  pthread_atfork(atfork, NULL, NULL);
  pthread_t t;
  pthread_create(&t, NULL, worker, NULL);
  glob++;
  pthread_join(t, NULL);
  // CHECK: ThreadSanitizer: data race
  // CHECK-NOT: ATFORK
  return 0;
}
