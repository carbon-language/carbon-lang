// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

char s[] = "abracadabra";

void *Thread0(void *p) {
  puts(s);
  return 0;
}

void *Thread1(void *p) {
  s[3] = 'z';
  return 0;
}

int main() {
  pthread_t th[2];
  pthread_create(&th[0], 0, Thread0, 0);
  pthread_create(&th[1], 0, Thread1, 0);
  pthread_join(th[0], 0);
  pthread_join(th[1], 0);
  fprintf(stderr, "DONE");
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: DONE

