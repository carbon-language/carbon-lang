// XFAIL: *
// RUN: %clangxx_asan -m64 -O2 %s -o %t
// RUN: %t 2>&1 | %symbolizer | c++filt | FileCheck %s

#include <stdio.h>

__attribute__((noinline))
char *Ident(char *x) {
  fprintf(stderr, "1: %p\n", x);
  return x;
}

__attribute__((noinline))
char *Func1() {
  char local;
  return Ident(&local);
}

__attribute__((noinline))
void Func2(char *x) {
  fprintf(stderr, "2: %p\n", x);
  *x = 1;
  // CHECK: {{WRITE of size 1 .* thread T0}}
  // CHECK: {{    #0.*Func2.*stack-use-after-return.cc:18}}
  // CHECK: {{is located in frame <.*Func1.*> of T0's stack}}
}

int main(int argc, char **argv) {
  Func2(Func1());
  return 0;
}
