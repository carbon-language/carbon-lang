// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// CHECK-NOT: WARNING
// CHECK: OK

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

pthread_mutex_t m;
pthread_cond_t c;
int x;

void *thr1(void *p) {
  pthread_mutex_lock(&m);
  pthread_cleanup_push((void(*)(void *arg))pthread_mutex_unlock, &m);
  while (x == 0)
    pthread_cond_wait(&c, &m);
  pthread_cleanup_pop(1);
  return 0;
}

int main() {
  pthread_t th;

  pthread_mutex_init(&m, 0);
  pthread_cond_init(&c, 0);

  pthread_create(&th, 0, thr1, 0);
  sleep(1);  // let it block on cond var
  pthread_cancel(th);

  pthread_join(th, 0);
  pthread_mutex_lock(&m);
  pthread_mutex_unlock(&m);
  fprintf(stderr, "OK\n");
}
