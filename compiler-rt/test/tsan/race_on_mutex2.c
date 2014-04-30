// RUN: %clang_tsan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <stddef.h>
#include <unistd.h>

void *Thread(void *x) {
  pthread_mutex_lock((pthread_mutex_t*)x);
  pthread_mutex_unlock((pthread_mutex_t*)x);
  return 0;
}

int main() {
  pthread_mutex_t Mtx;
  pthread_mutex_init(&Mtx, 0);
  pthread_t t;
  pthread_create(&t, 0, Thread, &Mtx);
  sleep(1);
  pthread_mutex_destroy(&Mtx);
  pthread_join(t, 0);
  return 0;
}

// CHECK:      WARNING: ThreadSanitizer: data race
