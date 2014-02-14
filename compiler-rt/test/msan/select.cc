// RUN: %clangxx_msan -m64 -O0 %s -o %t && not %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_msan -m64 -O1 %s -o %t && not %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_msan -m64 -O2 %s -o %t && not %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_msan -m64 -O3 %s -o %t && not %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <stdlib.h>
int main(int argc, char **argv) {
  int x;
  int *volatile p = &x;
  int z = *p ? 1 : 0;
  if (z)
    exit(0);
  // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // CHECK: {{#0 0x.* in main .*select.cc:}}[[@LINE-3]]

  // CHECK: SUMMARY: MemorySanitizer: use-of-uninitialized-value {{.*select.cc:.* main}}
  return 0;
}
