// RUN: %clangxx_tsan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

char *data = new char[10];
char *data1 = new char[10];
char *data2 = new char[10];

void *Thread1(void *x) {
  static volatile int size = 1;
  memcpy(data+5, data1, size);
  return NULL;
}

void *Thread2(void *x) {
  static volatile int size = 4;
  sleep(1);
  memcpy(data+3, data2, size);
  return NULL;
}

int main() {
  fprintf(stderr, "addr=%p\n", &data[5]);
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  return 0;
}

// CHECK: addr=[[ADDR:0x[0-9,a-f]+]]
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 1 at [[ADDR]] by thread T2:
// CHECK:     #0 memcpy
// CHECK:     #1 Thread2
// CHECK:   Previous write of size 1 at [[ADDR]] by thread T1:
// CHECK:     #0 memcpy
// CHECK:     #1 Thread1
