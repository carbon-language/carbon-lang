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
  int x;
  int *volatile p = &x;
  if (*p)
    exit(0);
  // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // CHECK: {{#0 0x.* in main .*stack-origin.cc:}}[[@LINE-3]]

  // CHECK-ORIGINS: Uninitialized value was created by an allocation of 'x' in the stack frame of function 'main'

  // CHECK: SUMMARY: MemorySanitizer: use-of-uninitialized-value {{.*stack-origin.cc:.* main}}
  return 0;
}
