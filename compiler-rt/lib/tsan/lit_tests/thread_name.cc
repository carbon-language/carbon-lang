// RUN: %clangxx_tsan -O1 %s -o %t && %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

extern "C" void AnnotateThreadName(const char *f, int l, const char *name);

int Global;

void *Thread1(void *x) {
  usleep(100*1000);
  AnnotateThreadName(__FILE__, __LINE__, "Thread1");
  Global++;
  return NULL;
}

void *Thread2(void *x) {
  AnnotateThreadName(__FILE__, __LINE__, "Thread2");
  Global--;
  return NULL;
}

int main() {
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Thread 1 'Thread1'
// CHECK:   Thread 2 'Thread2'

