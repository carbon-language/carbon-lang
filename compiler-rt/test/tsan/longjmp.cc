// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

// Longjmp assembly has not been implemented for mips64 or aarch64 yet
// XFAIL: mips64
// XFAIL: aarch64

#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>

int foo(jmp_buf env) {
  longjmp(env, 42);
}

int main() {
  jmp_buf env;
  if (setjmp(env) == 42) {
    printf("JUMPED\n");
    return 0;
  }
  foo(env);
  printf("FAILED\n");
  return 0;
}

// CHECK-NOT: FAILED
// CHECK: JUMPED
