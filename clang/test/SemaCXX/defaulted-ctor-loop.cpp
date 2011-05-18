// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

// WARNING: This test may recurse infinitely if failing.

struct foo;
struct bar {
  bar(foo&);
};
struct foo {
  bar b;
  foo()
    : b(b) // expected-warning{{field is uninitialized}}
  {}
};
