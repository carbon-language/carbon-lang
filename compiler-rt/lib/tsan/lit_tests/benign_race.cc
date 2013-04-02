// RUN: %clang_tsan -O1 %s -o %t && %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

int Global;
int WTFGlobal;

extern "C" {
void AnnotateBenignRaceSized(const char *f, int l,
                             void *mem, unsigned int size, const char *desc);
void WTFAnnotateBenignRaceSized(const char *f, int l,
                                void *mem, unsigned int size,
                                const char *desc);
}


void *Thread(void *x) {
  Global = 42;
  WTFGlobal = 142;
  return 0;
}

int main() {
  AnnotateBenignRaceSized(__FILE__, __LINE__,
                          &Global, sizeof(Global), "Race on Global");
  WTFAnnotateBenignRaceSized(__FILE__, __LINE__,
                             &WTFGlobal, sizeof(WTFGlobal),
                             "Race on WTFGlobal");
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  sleep(1);
  Global = 43;
  WTFGlobal = 143;
  pthread_join(t, 0);
  printf("OK\n");
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
