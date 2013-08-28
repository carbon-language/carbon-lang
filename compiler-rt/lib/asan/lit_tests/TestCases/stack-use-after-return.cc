// RUN: %clangxx_asan -fsanitize=use-after-return -O0 %s -o %t && \
// RUN:   not %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -fsanitize=use-after-return -O1 %s -o %t && \
// RUN:   not %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -fsanitize=use-after-return -O2 %s -o %t && \
// RUN:   not %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -fsanitize=use-after-return -O3 %s -o %t && \
// RUN:   not %t 2>&1 | FileCheck %s
// Regression test for a CHECK failure with small stack size and large frame.
// RUN: %clangxx_asan -fsanitize=use-after-return -O3 %s -o %t -DkSize=10000 && \
// RUN: (ulimit -s 65;  not %t) 2>&1 | FileCheck %s

#include <stdio.h>

#ifndef kSize
# define kSize 1
#endif

__attribute__((noinline))
char *Ident(char *x) {
  fprintf(stderr, "1: %p\n", x);
  return x;
}

__attribute__((noinline))
char *Func1() {
  char local[kSize];
  return Ident(local);
}

__attribute__((noinline))
void Func2(char *x) {
  fprintf(stderr, "2: %p\n", x);
  *x = 1;
  // CHECK: WRITE of size 1 {{.*}} thread T0
  // CHECK:     #0{{.*}}Func2{{.*}}stack-use-after-return.cc:[[@LINE-2]]
  // CHECK: is located in stack of thread T0 at offset
}

int main(int argc, char **argv) {
  Func2(Func1());
  return 0;
}
