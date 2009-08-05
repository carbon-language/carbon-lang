// RUN: clang-cc -fsyntax-only -verify %s

class Outer {
  int x;
  static int sx;

  class Inner {
    static char a[sizeof(x)]; // expected-error {{ invalid use of nonstatic data member 'x' }}
    static char b[sizeof(sx)]; // okay
  };
};
