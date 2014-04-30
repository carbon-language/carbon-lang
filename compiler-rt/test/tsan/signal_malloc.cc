// RUN: %clang_tsan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

static void handler(int, siginfo_t*, void*) {
  // CHECK: WARNING: ThreadSanitizer: signal-unsafe call inside of a signal
  // CHECK:     #0 malloc
  // CHECK:     #{{(1|2)}} handler(int, siginfo{{(_t)?}}*, void*) {{.*}}signal_malloc.cc:[[@LINE+2]]
  // CHECK: SUMMARY: ThreadSanitizer: signal-unsafe call inside of a signal{{.*}}handler
  volatile char *p = (char*)malloc(1);
  p[0] = 0;
  free((void*)p);
}

int main() {
  struct sigaction act = {};
  act.sa_sigaction = &handler;
  sigaction(SIGPROF, &act, 0);
  kill(getpid(), SIGPROF);
  sleep(1);
  return 0;
}

