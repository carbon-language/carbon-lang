// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>

int foo(sigjmp_buf env) {
  printf("env=%p\n", env);
  siglongjmp(env, 42);
}

int main() {
  sigjmp_buf env;
  printf("env=%p\n", env);
  if (sigsetjmp(env, 1) == 42) {
    printf("JUMPED\n");
    return 0;
  }
  foo(env);
  printf("FAILED\n");
  return 0;
}

// CHECK-NOT: FAILED
// CHECK: JUMPED
