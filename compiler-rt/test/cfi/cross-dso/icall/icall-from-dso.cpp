// RUN: %clangxx_cfi_dso -DSHARED_LIB %s -fPIC -shared -o %t-so.so
// RUN: %clangxx_cfi_dso %s -o %t %t-so.so && %expect_crash %t 2>&1 | FileCheck %s

#include <stdio.h>

#ifdef SHARED_LIB
void g();
void f() {
  // CHECK: =1=
  fprintf(stderr, "=1=\n");
  ((void (*)(void))g)();
  // CHECK: =2=
  fprintf(stderr, "=2=\n");
  ((void (*)(int))g)(42); // UB here
  // CHECK-NOT: =3=
  fprintf(stderr, "=3=\n");
}
#else
void f();
void g() {
}

int main() {
  f();
}
#endif
