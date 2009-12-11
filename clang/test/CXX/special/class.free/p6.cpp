// RUN: clang-cc -fsyntax-only -verify %s
#include <stddef.h>

struct A {
  void operator delete(void*) {
    (void)this; // expected-error {{invalid use of 'this' outside of a nonstatic member function}}
  }
  void operator delete[](void*) {
    (void)this; // expected-error {{invalid use of 'this' outside of a nonstatic member function}}
  }
};
