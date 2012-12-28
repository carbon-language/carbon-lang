// Test blacklist functionality for TSan.

// RUN: %clangxx_tsan -O1 %s \
// RUN:   -fsanitize-blacklist=%p/Helpers/blacklist.txt \
// RUN:   -o %t && %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>

int Global;

void *Thread1(void *x) {
  Global++;
  return NULL;
}

void *Blacklisted_Thread2(void *x) {
  Global--;
  return NULL;
}

int main() {
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Blacklisted_Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  printf("PASS\n");
  return 0;
}

// CHECK-NOT: ThreadSanitizer: data race
