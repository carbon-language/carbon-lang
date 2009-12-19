// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR5741
struct A {
  struct B { };
  struct C;
};

struct A::C : B { };
