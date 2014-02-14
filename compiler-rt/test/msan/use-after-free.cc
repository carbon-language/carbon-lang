// RUN: %clangxx_msan -m64 -O0 %s -o %t && not %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_msan -m64 -O1 %s -o %t && not %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_msan -m64 -O2 %s -o %t && not %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_msan -m64 -O3 %s -o %t && not %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_msan -fsanitize-memory-track-origins -m64 -O0 %s -o %t && not %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-ORIGINS < %t.out
// RUN: %clangxx_msan -fsanitize-memory-track-origins -m64 -O1 %s -o %t && not %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-ORIGINS < %t.out
// RUN: %clangxx_msan -fsanitize-memory-track-origins -m64 -O2 %s -o %t && not %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-ORIGINS < %t.out
// RUN: %clangxx_msan -fsanitize-memory-track-origins -m64 -O3 %s -o %t && not %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-ORIGINS < %t.out

#include <stdlib.h>
int main(int argc, char **argv) {
  int *volatile p = (int *)malloc(sizeof(int));
  *p = 42;
  free(p);

  if (*p)
    exit(0);
  // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // CHECK: {{#0 0x.* in main .*use-after-free.cc:}}[[@LINE-3]]

  // CHECK-ORIGINS: Uninitialized value was created by a heap allocation
  // CHECK-ORIGINS: {{#0 0x.* in .*free}}
  // CHECK-ORIGINS: {{#1 0x.* in main .*use-after-free.cc:}}[[@LINE-9]]
  return 0;
}
