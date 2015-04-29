// RUN: %clang_tsan -O1 %s -o %t && TSAN_OPTIONS="flush_memory_ms=1 memory_limit_mb=1" %run %t 2>&1 | FileCheck %s
#include "test.h"
#include <sys/types.h>
#include <unistd.h>
#include <time.h>

// Test that setuid call works in presence of stoptheworld.

int main() {
  struct timespec tp0, tp1;
  clock_gettime(CLOCK_MONOTONIC, &tp0);
  clock_gettime(CLOCK_MONOTONIC, &tp1);
  while (tp1.tv_sec - tp0.tv_sec < 3) {
    clock_gettime(CLOCK_MONOTONIC, &tp1);
    setuid(0);
  }
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: DONE
