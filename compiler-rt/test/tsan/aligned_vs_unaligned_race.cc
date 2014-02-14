// RUN: %clangxx_tsan -O1 %s -o %t && %t 2>&1 | FileCheck %s
// Race between an aligned access and an unaligned access, which
// touches the same memory region.
// This is a real race which is not detected by tsan.
// https://code.google.com/p/thread-sanitizer/issues/detail?id=17
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>

uint64_t Global[2];

void *Thread1(void *x) {
  Global[1]++;
  return NULL;
}

void *Thread2(void *x) {
  char *p1 = reinterpret_cast<char *>(&Global[0]);
  uint64_t *p4 = reinterpret_cast<uint64_t *>(p1 + 1);
  (*p4)++;
  return NULL;
}

int main() {
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  printf("Pass\n");
  // CHECK-NOT: ThreadSanitizer: data race
  // CHECK: Pass
  return 0;
}
