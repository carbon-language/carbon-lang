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
};
