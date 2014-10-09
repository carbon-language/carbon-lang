// Test that blacklisted functions are still contained in the stack trace.

// RUN: echo "fun:*Blacklisted_Thread2*" > %t.blacklist
// RUN: echo "fun:*CallTouchGlobal*" >> %t.blacklist

// RUN: %clangxx_tsan -O1 %s -fsanitize-blacklist=%t.blacklist -o %t
// RUN: %deflake %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

int Global;

void *Thread1(void *x) {
  sleep(1);
  // CHECK: ThreadSanitizer: data race
  // CHECK: Write of size 4
  // CHECK: #0 Thread1{{.*}}blacklist2.cc:[[@LINE+1]]
  Global++;
  return NULL;
}

void TouchGlobal() {
  // CHECK: Previous write of size 4
  // CHECK: #0 TouchGlobal{{.*}}blacklist2.cc:[[@LINE+1]]
  Global--;
}

void CallTouchGlobal() {
  // CHECK: #1 CallTouchGlobal{{.*}}blacklist2.cc:[[@LINE+1]]
  TouchGlobal();
}

void *Blacklisted_Thread2(void *x) {
  Global--;
  // CHECK: #2 Blacklisted_Thread2{{.*}}blacklist2.cc:[[@LINE+1]]
  CallTouchGlobal();
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
