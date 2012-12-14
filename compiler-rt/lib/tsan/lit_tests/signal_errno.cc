// RUN: %clang_tsan -O1 %s -o %t && %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

pthread_t mainth;
volatile int done;

static void handler(int, siginfo_t *s, void *c) {
  errno = 1;
  done = 1;
}

static void* sendsignal(void *p) {
  pthread_kill(mainth, SIGPROF);
  return 0;
}

int main() {
  mainth = pthread_self();
  struct sigaction act = {};
  act.sa_sigaction = &handler;
  sigaction(SIGPROF, &act, 0);
  pthread_t th;
  pthread_create(&th, 0, sendsignal, 0);
  while (done == 0) {
    volatile char *p = (char*)malloc(1);
    p[0] = 0;
    free((void*)p);
    pthread_yield();
  }
  pthread_join(th, 0);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: signal handler spoils errno
// CHECK:     #0 handler(int, siginfo*, void*) {{.*}}signal_errno.cc

