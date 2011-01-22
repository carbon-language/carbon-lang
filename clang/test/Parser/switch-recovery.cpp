// RUN: %clang_cc1 -fsyntax-only -verify %s

// <rdar://problem/7971948>
struct A {};
struct B {
  void foo(int b) {
    switch (a) { // expected-error{{use of undeclared identifier 'a'}}
    default:
      return;
    }
    
    switch (b) {
    case 17 // expected-error{{expected ':' after 'case'}}
      break;

    default // expected-error{{expected ':' after 'default'}}
      return;
    }
  }

  void test2() {
    enum X { Xa, Xb } x;

    switch (x) { // expected-warning {{enumeration value 'Xb' not handled in switch}}
    case Xa; // expected-error {{expected ':' after 'case'}}
      break;
    }

    switch (x) {
    default; // expected-error {{expected ':' after 'default'}}
      break;
    }
  }
};
