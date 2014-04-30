// RUN: %clangxx_tsan -O1 %s -o %t && %run %t | FileCheck %s
// Regtest for https://code.google.com/p/thread-sanitizer/issues/detail?id=40
// This is a correct program and tsan should not report a race.
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
int g;
__attribute__((noinline))
int foo(int cond) {
  if (cond)
    return g;
  return 0;
}
void *Thread1(void *p) {
  long res = foo((long)p);
  sleep(1);
  return (void*) res;
}

int main() {
  pthread_t t;
  pthread_create(&t, 0, Thread1, 0);
  g = 1;
  pthread_join(t, 0);
  printf("PASS\n");
  // CHECK: PASS
}
