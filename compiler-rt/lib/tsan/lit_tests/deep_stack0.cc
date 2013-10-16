// RUN: %clangxx_tsan -O1 %s -o %t && not %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

volatile int X;
volatile int N;
void (*volatile F)();

static void foo() {
  if (--N == 0)
    X = 42;
  else
    F();
}

void *Thread(void *p) {
  sleep(1);
  F();
  return 0;
}

int main() {
  N = 50000;
  F = foo;
  pthread_t t;
  pthread_attr_t a;
  pthread_attr_init(&a);
  pthread_attr_setstacksize(&a, N * 256 + (1 << 20));
  pthread_create(&t, &a, Thread, 0);
  X = 43;
  pthread_join(t, 0);
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:    #100 foo
// We must output suffucuently large stack (at least 100 frames)

