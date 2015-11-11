// Stress test recovery mode with many threads.
//
// RUN: %clangxx_asan -fsanitize-recover=address -pthread %s -o %t
//
// RUN: env ASAN_OPTIONS=halt_on_error=false:max_errors=1000 %run %t 1 10 >1.txt 2>&1
// RUN: FileCheck %s < 1.txt
// RUN: [ $(wc -l < 1.txt) -gt 1 ]
//
// RUN: env ASAN_OPTIONS=halt_on_error=false:max_errors=1000 %run %t 10 20 >10.txt 2>&1
// RUN: FileCheck %s < 10.txt
// This one is racy although very unlikely to fail:
// RUN: [ $(wc -l < 10.txt) -gt 1 ]
// Collisions are highly unlikely but still possible so we need the alternative:
// RUN: FileCheck --check-prefix=CHECK-COLLISION %s < 1.txt || FileCheck --check-prefix=CHECK-NO-COLLISION %s < 1.txt
//
// REQUIRES: stable-runtime

#define _POSIX_C_SOURCE 200112  // rand_r

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#include <sanitizer/asan_interface.h>

size_t nthreads = 10;
size_t niter = 10;

void *run(void *arg) {
  unsigned seed = (unsigned)(size_t)arg;

  volatile char tmp[2];
  __asan_poison_memory_region(&tmp, sizeof(tmp)); 

  for (size_t i = 0; i < niter; ++i) {
    struct timespec delay = { 0, rand_r(&seed) * 1000000 };
    nanosleep(&delay, 0);

    // Expect error collisions here
    // CHECK: AddressSanitizer: use-after-poison
    volatile int idx = 0;
    tmp[idx] = 0;
  }

  return 0;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "Syntax: %s nthreads niter\n", argv[0]);
    exit(1);
  }

  nthreads = (size_t)strtoul(argv[1], 0, 0);
  niter = (size_t)strtoul(argv[2], 0, 0);

  pthread_t *tids = new pthread_t[nthreads];

  for (size_t i = 0; i < nthreads; ++i) {
    if (0 != pthread_create(&tids[i], 0, run, (void *)i)) {
      fprintf(stderr, "Failed to create thread\n");
      exit(1);
    }
  }

  for (size_t i = 0; i < nthreads; ++i) {
    if (0 != pthread_join(tids[i], 0)) {
      fprintf(stderr, "Failed to join thread\n");
      exit(1);
    }
  }

  // CHECK-COLLISION: AddressSanitizer: nested bug in the same thread, aborting
  // CHECK-NO-COLLISION: All threads terminated
  printf("All threads terminated\n");

  delete [] tids;

  return 0;
}
