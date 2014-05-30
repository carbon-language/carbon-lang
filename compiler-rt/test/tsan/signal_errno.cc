// RUN: %clang_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

pthread_t mainth;
volatile int done;

static void MyHandler(int, siginfo_t *s, void *c) {
  errno = 1;
  done = 1;
}

static void* sendsignal(void *p) {
  sleep(1);
  pthread_kill(mainth, SIGPROF);
  return 0;
}

static __attribute__((noinline)) void loop() {
  while (done == 0) {
    volatile char *p = (char*)malloc(1);
    p[0] = 0;
    free((void*)p);
    pthread_yield();
  }
}

int main() {
  mainth = pthread_self();
  struct sigaction act = {};
  act.sa_sigaction = &MyHandler;
  sigaction(SIGPROF, &act, 0);
  pthread_t th;
  pthread_create(&th, 0, sendsignal, 0);
  loop();
  pthread_join(th, 0);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: signal handler spoils errno
// CHECK:     #0 MyHandler(int, siginfo{{(_t)?}}*, void*) {{.*}}signal_errno.cc
// CHECK:     main
// CHECK: SUMMARY: ThreadSanitizer: signal handler spoils errno{{.*}}MyHandler

