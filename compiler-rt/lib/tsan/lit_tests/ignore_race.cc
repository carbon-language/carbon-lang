// RUN: %clang_tsan -O1 %s -o %t && %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

int Global;

extern "C" void AnnotateIgnoreWritesBegin(const char *f, int l);
extern "C" void AnnotateIgnoreWritesEnd(const char *f, int l);
extern "C" void AnnotateIgnoreReadsBegin(const char *f, int l);
extern "C" void AnnotateIgnoreReadsEnd(const char *f, int l);

void *Thread(void *x) {
  AnnotateIgnoreWritesBegin(__FILE__, __LINE__);
  AnnotateIgnoreReadsBegin(__FILE__, __LINE__);
  Global = 42;
  AnnotateIgnoreReadsEnd(__FILE__, __LINE__);
  AnnotateIgnoreWritesEnd(__FILE__, __LINE__);
  return 0;
}

int main() {
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  usleep(100000);
  Global = 43;
  pthread_join(t, 0);
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
