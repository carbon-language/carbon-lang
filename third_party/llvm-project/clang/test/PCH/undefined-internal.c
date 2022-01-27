// RUN: %clang_cc1 -emit-pch %s -o %t
// RUN: %clang_cc1 -include-pch %t %s -verify
#ifndef HEADER_H
#define HEADER_H
static void f();
static void g();
void h() {
  f();
  g();
}
#else
static void g() {}
// expected-warning@5{{function 'f' has internal linkage but is not defined}}
// expected-note@8{{used here}}
#endif
