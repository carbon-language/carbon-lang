// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

struct A {
  virtual void f();
  virtual void h();
};

struct B : A {
  [[override]] virtual void f();
  [[override]] void g(); // expected-error {{'override' attribute only applies to virtual method types}}
  [[override, override]] void h(); // expected-error {{'override' attribute cannot be repeated}}
};
