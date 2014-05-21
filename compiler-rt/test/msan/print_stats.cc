// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -m64 -g %s -o %t 
// RUN: %run %t 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK --check-prefix=CHECK-NOSTATS %s
// RUN: MSAN_OPTIONS=print_stats=1 %run %t 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK --check-prefix=CHECK-NOSTATS %s

// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -m64 -g -DPOSITIVE=1 %s -o %t 
// RUN: not %run %t 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK --check-prefix=CHECK-NOSTATS %s
// RUN: MSAN_OPTIONS=print_stats=1 not %run %t 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK --check-prefix=CHECK-STATS %s

// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -m64 -g -DPOSITIVE=1 -mllvm -msan-keep-going=1 %s -o %t 
// RUN: not %run %t 2>&1 | \
// RUN:  FileCheck --check-prefix=CHECK --check-prefix=CHECK-NOSTATS --check-prefix=CHECK-KEEPGOING %s
// RUN: MSAN_OPTIONS=print_stats=1 not %run %t 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK --check-prefix=CHECK-STATS --check-prefix=CHECK-KEEPGOING %s

#include <stdio.h>
int main(int argc, char **argv) {
  int x;
  int *volatile p = &x;
  fprintf(stderr, "TEST\n");
#ifdef POSITIVE
  return *p;
#else
  return 0;
#endif
}

// CHECK: TEST

// CHECK-KEEPGOING: MemorySanitizer: 1 warnings reported.

// CHECK-STATS: Unique heap origins:
// CHECK-STATS: Stack depot allocated bytes:
// CHECK-STATS: Unique origin histories:
// CHECK-STATS: History depot allocated bytes:

// CHECK-NOSTATS-NOT: Unique heap origins:
// CHECK-NOSTATS-NOT: Stack depot allocated bytes:
// CHECK-NOSTATS-NOT: Unique origin histories:
// CHECK-NOSTATS-NOT: History depot allocated bytes:
