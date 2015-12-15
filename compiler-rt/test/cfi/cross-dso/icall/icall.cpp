// RUN: %clangxx_cfi_dso -DSHARED_LIB %s -fPIC -shared -o %t-so.so
// RUN: %clangxx_cfi_dso %s -o %t %t-so.so && %expect_crash %t 2>&1 | FileCheck %s

#include <stdio.h>

#ifdef SHARED_LIB
void f() {
}
#else
void f();
int main() {
  // CHECK: =1=
  fprintf(stderr, "=1=\n");
  ((void (*)(void))f)();
  // CHECK: =2=
  fprintf(stderr, "=2=\n");
  ((void (*)(int))f)(42); // UB here
  // CHECK-NOT: =3=
  fprintf(stderr, "=3=\n");
}
#endif
