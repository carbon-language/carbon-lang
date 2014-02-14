// RUN: %clang_tsan -O1 %s -o %t && not %t 2>&1 | FileCheck %s
// CHECK: ThreadSanitizer: data race
// CHECK: pthread_cond_signal

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

struct Ctx {
  pthread_mutex_t m;
  pthread_cond_t c;
  bool done;
};

void *thr(void *p) {
  Ctx *c = (Ctx*)p;
  pthread_mutex_lock(&c->m);
  c->done = true;
  pthread_mutex_unlock(&c->m);
  pthread_cond_signal(&c->c);
  return 0;
}

int main() {
  Ctx *c = new Ctx();
  pthread_mutex_init(&c->m, 0);
  pthread_cond_init(&c->c, 0);
  pthread_t th;
  pthread_create(&th, 0, thr, c);
  pthread_mutex_lock(&c->m);
  while (!c->done)
    pthread_cond_wait(&c->c, &c->m);
  pthread_mutex_unlock(&c->m);
  delete c;
  pthread_join(th, 0);
}
