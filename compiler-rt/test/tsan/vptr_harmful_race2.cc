// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <unistd.h>

struct A {
  A() {
    sem_init(&sem_, 0, 0);
  }
  virtual void F() {
  }
  void Done() {
    sem_post(&sem_);
  }
  virtual ~A() {
    sem_wait(&sem_);
    sem_destroy(&sem_);
  }
  sem_t sem_;
};

struct B : A {
  virtual void F() {
  }
  virtual ~B() { }
};

static A *obj = new B;

void *Thread1(void *x) {
  sleep(1);
  obj->F();
  obj->Done();
  return NULL;
}

void *Thread2(void *x) {
  delete obj;
  return NULL;
}

int main() {
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
}

// CHECK: WARNING: ThreadSanitizer: data race on vptr
