// RUN: %clangxx_tsan -O1 %s -o %t && not %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

extern "C" void AnnotateThreadName(const char *f, int l, const char *name);

int Global;

void *Thread1(void *x) {
  sleep(1);
  AnnotateThreadName(__FILE__, __LINE__, "Thread1");
  Global++;
  return NULL;
}

void *Thread2(void *x) {
#if SANITIZER_LINUX && __GLIBC_PREREQ(2, 12)
  pthread_setname_np(pthread_self(), "Thread2");
#else
  AnnotateThreadName(__FILE__, __LINE__, "Thread2");
#endif
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
// CHECK:   Thread T1 'Thread1'
// CHECK:   Thread T2 'Thread2'

