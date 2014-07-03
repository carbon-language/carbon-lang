// RUN: %clangxx_msan -m64 -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: %clangxx_msan -m64 -O1 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_msan -m64 -O2 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_msan -m64 -O3 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// Test that (no_sanitize_memory) functions propagate shadow.

// Note that at -O0 there is no report, because 'x' in 'f' is spilled to the
// stack, and then loaded back as a fully initialiazed value (due to
// no_sanitize_memory attribute).

#include <stdlib.h>
#include <stdio.h>

__attribute__((noinline))
__attribute__((no_sanitize_memory))
int f(int x) {
  return x;
}

int main(void) {
  int x;
  int * volatile p = &x;
  int y = f(*p);
  // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // CHECK: {{#0 0x.* in main .*no_sanitize_memory_prop.cc:}}[[@LINE+1]]
  if (y)
    exit(0);
  return 0;
}
