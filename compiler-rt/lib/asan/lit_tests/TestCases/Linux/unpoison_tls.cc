// Test that TLS is unpoisoned on thread death.
// REQUIRES: x86_64-supported-target,i386-supported-target

// RUN: %clangxx_asan -O1 %s -o %t && %t 2>&1

#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <sanitizer/asan_interface.h>

__thread int64_t tls_var[2];

volatile int64_t *p_tls_var;

void *first(void *arg) {
  ASAN_POISON_MEMORY_REGION(&tls_var, sizeof(tls_var));
  p_tls_var = tls_var;
  return 0;
}

void *second(void *arg) {
  assert(tls_var == p_tls_var);
  *p_tls_var = 1;
  return 0;
}

int main(int argc, char *argv[]) {
  pthread_t p;
  assert(0 == pthread_create(&p, 0, first, 0));
  assert(0 == pthread_join(p, 0));
  assert(0 == pthread_create(&p, 0, second, 0));
  assert(0 == pthread_join(p, 0));
  return 0;
}
