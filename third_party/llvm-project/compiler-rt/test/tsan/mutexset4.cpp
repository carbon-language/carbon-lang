// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

int Global;
pthread_mutex_t mtx1;
pthread_mutex_t mtx2;

void *Thread1(void *x) {
  pthread_mutex_lock(&mtx1);
  pthread_mutex_lock(&mtx2);
  Global++;
  pthread_mutex_unlock(&mtx2);
  pthread_mutex_unlock(&mtx1);
  barrier_wait(&barrier);
  return NULL;
}

void *Thread2(void *x) {
  barrier_wait(&barrier);
  Global--;
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  // CHECK: WARNING: ThreadSanitizer: data race
  // CHECK:   Write of size 4 at {{.*}} by thread T2:
  // CHECK:   Previous write of size 4 at {{.*}} by thread T1
  // CHECK:                 (mutexes: write [[M1:M[0-9]+]], write [[M2:M[0-9]+]]):
  // CHECK:   Mutex [[M1]] (0x{{.*}}) created at:
  // CHECK:     #0 pthread_mutex_init
  // CHECK:     #1 main {{.*}}mutexset4.cpp:[[@LINE+4]]
  // CHECK:   Mutex [[M2]] (0x{{.*}}) created at:
  // CHECK:     #0 pthread_mutex_init
  // CHECK:     #1 main {{.*}}mutexset4.cpp:[[@LINE+2]]
  pthread_mutex_init(&mtx1, 0);
  pthread_mutex_init(&mtx2, 0);
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  pthread_mutex_destroy(&mtx1);
  pthread_mutex_destroy(&mtx2);
}
