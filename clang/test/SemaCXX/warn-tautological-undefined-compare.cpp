// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wtautological-undefined-compare %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wno-tautological-compare -Wtautological-undefined-compare %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wtautological-compare %s

void test1(int &x) {
  if (x == 1) { }
  if (&x == 0) { }
  // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to false}}
  if (&x != 0) { }
  // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to true}}
}

class test2 {
  test2() : x(y) {}

  void foo() {
    if (this == 0) { }
    // expected-warning@-1{{'this' pointer cannot be null in well-defined C++ code; comparison may be assumed to always evaluate to false}}
    if (this != 0) { }
    // expected-warning@-1{{'this' pointer cannot be null in well-defined C++ code; comparison may be assumed to always evaluate to true}}
  }

  void bar() {
    if (x == 1) { }
    if (&x == 0) { }
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to false}}
    if (&x != 0) { }
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to true}}
  }

  int &x;
  int y;
};
