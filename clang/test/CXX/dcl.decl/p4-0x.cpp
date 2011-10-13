// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct X {
  void f() &;
  void g() &&;
};

void (X::*pmf)() & = &X::f;
