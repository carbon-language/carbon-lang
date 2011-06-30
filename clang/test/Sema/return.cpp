// RUN: %clang %s -fsyntax-only -Wreturn-type

struct C {
  C() {
    return 42; // expected-warning {{constructor 'C' should not return a value}}
  }
  ~C() {
    return 42; // expected-warning {{destructor '~C' should not return a value}}
  }
};
