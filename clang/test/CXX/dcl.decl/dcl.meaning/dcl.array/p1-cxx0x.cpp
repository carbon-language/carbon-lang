// RUN: clang-cc -fsyntax-only -verify %s -std=c++0x

void f() {
  int b[5];
  auto a[5] = b; // expected-error{{'a' declared as array of 'auto'}}
}
