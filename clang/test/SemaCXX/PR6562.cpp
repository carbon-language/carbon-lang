// RUN: %clang_cc1 -fsyntax-only -verify %s

struct X { ~X(); };
template <typename T>
struct A {
  struct B { X x; };
  struct C : public B {
    C() : B() { }
  };
};
