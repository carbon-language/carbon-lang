// RUN: %clangxx_tsan -O1 %s -o %t && %t 2>&1 | FileCheck %s -check-prefix=CHECK-DIE
// RUN: %clangxx_tsan -O1 %s -o %t && TSAN_OPTIONS="die_after_fork=0" %t 2>&1 | FileCheck %s -check-prefix=CHECK-NODIE
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

static void *sleeper(void *p) {
  sleep(10);
  return 0;
}

int main() {
  pthread_t th;
  pthread_create(&th, 0, sleeper, 0);
  switch (fork()) {
  default:  // parent
    while (wait(0) < 0) {}
    break;
  case 0:  // child
    {
      pthread_t th2;
      pthread_create(&th2, 0, sleeper, 0);
      exit(0);
      break;
    }
  case -1:  // error
    fprintf(stderr, "failed to fork (%d)\n", errno);
    exit(1);
  }
  fprintf(stderr, "OK\n");
}

// CHECK-DIE: ThreadSanitizer: starting new threads after muti-threaded fork is not supported
// CHECK-DIE: OK

// CHECK-NODIE-NOT: ThreadSanitizer: starting new threads after muti-threaded fork is not supported
// CHECK-NODIE: OK

