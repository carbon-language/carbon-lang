// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
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
