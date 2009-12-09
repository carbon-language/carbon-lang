// RUN: clang-cc -fsyntax-only -verify -Wmissing-prototypes %s

void f() { } // expected-warning {{no previous prototype for function 'f'}}

namespace NS {
  void f() { } // expected-warning {{no previous prototype for function 'f'}}
}

namespace {
  // Should not warn about anonymous namespaces
  void f() { }
}

struct A {
  // Should not warn about member functions.
  void f() { }
};

inline void g() { }