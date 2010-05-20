// RUN: %clang_cc1 -fsyntax-only -verify %s

// <rdar://problem/7971948>
struct A {};
struct B {
  void foo() {
    switch (a) { // expected-error{{use of undeclared identifier 'a'}}
    default:
      return;
    }
  }
};
