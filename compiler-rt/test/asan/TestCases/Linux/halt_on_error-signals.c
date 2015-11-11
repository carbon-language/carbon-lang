// Test interaction of Asan recovery mode with asynch signals.
//
// RUN: %clang_asan -fsanitize-recover=address %s -o %t
//
// RUN: rm -f %t.log
// RUN: env ASAN_OPTIONS=halt_on_error=false %run %t 1000 >%t.log 2>&1 || true
// RUN: FileCheck %s < %t.log
// Collision will almost always get triggered but we still need to check the unlikely case:
// RUN: FileCheck --check-prefix=CHECK-COLLISION %s < %t.log || FileCheck --check-prefix=CHECK-NO-COLLISION %s < %t.log
//
// REQUIRES: stable-runtime

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>

#include <sanitizer/asan_interface.h>

void random_sleep(unsigned *seed) {
  struct timespec delay = { 0, rand_r(seed) * 1000000 };
  nanosleep(&delay, 0);
}

volatile char bad[2] = {1, };

void error() {
  // CHECK-COLLISION: AddressSanitizer: nested bug in the same thread, aborting
  // CHECK: AddressSanitizer: use-after-poison
  volatile int idx = 0;
  bad[idx] = 0;
}

size_t niter = 10;
pthread_t sender_tid, receiver_tid;

void *sender(void *arg) {
  unsigned seed = 0;
  for (size_t i = 0; i < niter; ++i) {
    random_sleep(&seed);
    pthread_kill(receiver_tid, SIGUSR1);
  }
  return 0;
}

void handler(int sig) {
  // Expect error collisions here
  error();
}

void *receiver(void *arg) {
  unsigned seed = 1;
  for (size_t i = 0; i < niter; ++i) {
    random_sleep(&seed);
    // And here
    error();
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Syntax: %s niter\n", argv[0]);
    exit(1);
  }

  niter = (size_t)strtoul(argv[1], 0, 0);

  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = handler;
  sa.sa_flags = SA_NODEFER; // Enable nested handlers to add more stress
  if (0 != sigaction(SIGUSR1, &sa, 0)) {
    fprintf(stderr, "Failed to set sighandler\n");
    exit(1);
  }

  __asan_poison_memory_region(&bad, sizeof(bad)); 

  if (0 != pthread_create(&receiver_tid, 0, receiver, 0)) {
    fprintf(stderr, "Failed to start receiver thread\n");
    exit(1);
  }

  if (0 != pthread_create(&sender_tid, 0, sender, 0)) {
    fprintf(stderr, "Failed to start sender thread\n");
    exit(1);
  }

  if (0 != pthread_join(receiver_tid, 0)) {
    fprintf(stderr, "Failed to wait receiver thread\n");
    exit(1);
  }

  if (0 != pthread_join(sender_tid, 0)) {
    fprintf(stderr, "Failed to wait sender thread\n");
    exit(1);
  }

  // CHECK-NO-COLLISION: All threads terminated
  printf("All threads terminated\n");

  return 0;
}
