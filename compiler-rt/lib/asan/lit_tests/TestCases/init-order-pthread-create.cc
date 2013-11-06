// Check that init-order checking is properly disabled if pthread_create is
// called.

// RUN: %clangxx_asan %s %p/Helpers/init-order-pthread-create-extra.cc -o %t
// RUN: ASAN_OPTIONS=strict_init_order=true %t

#include <stdio.h>
#include <pthread.h>

void *run(void *arg) {
  return arg;
}

void *foo(void *input) {
  pthread_t t;
  pthread_create(&t, 0, run, input);
  void *res;
  pthread_join(t, &res);
  return res;
}

void *bar(void *input) {
  return input;
}

void *glob = foo((void*)0x1234);
extern void *glob2;

int main() {
  printf("%p %p\n", glob, glob2);
  return 0;
}
