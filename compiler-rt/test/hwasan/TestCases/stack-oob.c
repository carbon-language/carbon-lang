// RUN: %clang_hwasan_oldrt -DSIZE=2 -O0 %s -o %t && %run %t
// RUN: %clang_hwasan -DSIZE=2 -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hwasan_oldrt -DSIZE=15 -O0 %s -o %t && %run %t
// RUN: %clang_hwasan -DSIZE=15 -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hwasan_oldrt -DSIZE=16 -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hwasan -DSIZE=16 -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hwasan -DSIZE=64 -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hwasan -DSIZE=0x1000 -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

// REQUIRES: stable-runtime

// Stack short granules are currently not implemented on x86.
// XFAIL: x86_64

#include <stdlib.h>
#include <sanitizer/hwasan_interface.h>

__attribute__((noinline))
int f() {
  char z[SIZE];
  char *volatile p = z;
  return p[SIZE];
}

int main() {
  f();
  // CHECK: READ of size 1 at
  // CHECK: #0 {{.*}} in f{{.*}}stack-oob.c:[[@LINE-6]]

  // CHECK-NOT: Cause: global-overflow
  // CHECK: Cause: stack tag-mismatch
  // CHECK: is located in stack of threa
  // CHECK-NOT: Cause: global-overflow

  // CHECK: SUMMARY: HWAddressSanitizer: tag-mismatch {{.*}} in f
}
