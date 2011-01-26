// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

struct X {
  void f() &;
  void g() &&;
};

void (X::*pmf)() & = &X::f;
