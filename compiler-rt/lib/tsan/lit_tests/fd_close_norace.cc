// RUN: %clangxx_tsan -O1 %s -o %t && %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

void *Thread1(void *x) {
  int f = open("/dev/random", O_RDONLY);
  close(f);
  return NULL;
}

void *Thread2(void *x) {
  sleep(1);
  int f = open("/dev/random", O_RDONLY);
  close(f);
  return NULL;
}

int main() {
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race


