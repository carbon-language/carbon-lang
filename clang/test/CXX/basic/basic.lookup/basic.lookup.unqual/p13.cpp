// RUN: clang-cc -fsyntax-only -verify %s

struct S {
  static const int f0 = 0;
  static int f1;
};

int S::f1 = f0;
