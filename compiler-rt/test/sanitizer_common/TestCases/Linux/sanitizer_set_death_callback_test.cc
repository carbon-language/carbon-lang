// RUN: %clangxx -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// Check __sanitizer_set_death_callback. Not all sanitizers implement it yet.
// XFAIL: lsan
// XFAIL: tsan

#include <sanitizer/common_interface_defs.h>
#include <stdio.h>
#include <pthread.h>

volatile char *zero = 0;

void Death() {
  fprintf(stderr, "DEATH CALLBACK EXECUTED\n");
}
// CHECK: DEATH CALLBACK EXECUTED

int global[10];
volatile char *sink;

void *Thread(void *x) {
  global[0]++;
  return x;
}

__attribute__((noinline))
void MaybeInit(int *uninitialized) {
  if (zero)
    *uninitialized = 1;
}

__attribute__((noinline))
void Leak() {
  sink = new char[100];  // trigger lsan report.
}

int main(int argc, char **argv) {
  int uninitialized;
  __sanitizer_set_death_callback(Death);
  MaybeInit(&uninitialized);
  if (uninitialized)  // trigger msan report.
    global[0] = 77;
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  global[0]++;           // trigger tsan report.
  pthread_join(t, 0);
  global[argc + 10]++;   // trigger asan report.
  Leak();
  sink = 0;
}
