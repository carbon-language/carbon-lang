// RUN: %clang_cc1 -fsyntax-only -verify %s

class Outer {
  int x;
  static int sx;

  // C++0x will likely relax this rule in this specific case, but
  // we'll still need to enforce it in C++03 mode.  See N2253 (or
  // successor).
  class Inner {
    static char a[sizeof(x)]; // expected-error {{ invalid use of nonstatic data member 'x' }}
    static char b[sizeof(sx)]; // okay
  };
};
