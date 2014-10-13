// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

struct A {
  virtual void F() {
  }

  virtual ~A() {
  }
};

struct B : A {
  virtual void F() {
  }
};

void *Thread(void *x) {
  sleep(1);
  ((A*)x)->F();
  return 0;
}

int main() {
  A *obj = new B;
  pthread_t t;
  pthread_create(&t, 0, Thread, obj);
  delete obj;
  pthread_join(t, 0);
}

// CHECK: WARNING: ThreadSanitizer: heap-use-after-free (virtual call vs free)

