// RUN: %clang_cc1 -verify -std=c++11 %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -x c++ -std=c++11 -fixit %t
// RUN: %clang_cc1 -Wall -pedantic -x c++ -std=c++11 %t

/* This is a test of the various code modification hints that only
   apply in C++0x. */
struct A {
  explicit operator int(); // expected-note{{conversion to integral type}}
};

void x() {
  switch(A()) { // expected-error{{explicit conversion to}}
  }
}

using ::T = void; // expected-error {{name defined in alias declaration must be an identifier}}
using typename U = void; // expected-error {{name defined in alias declaration must be an identifier}}
using typename ::V = void; // expected-error {{name defined in alias declaration must be an identifier}}

namespace Constexpr {
  extern constexpr int a; // expected-error {{must be a definition}}
  // -> extern const int a;

  extern constexpr int *b; // expected-error {{must be a definition}}
  // -> extern int *const b;

  extern constexpr int &c; // expected-error {{must be a definition}}
  // -> extern int &b;

  extern constexpr const int d; // expected-error {{must be a definition}}
  // -> extern const int d;

  int z;
  constexpr int a = 0;
  constexpr int *b = &z;
  constexpr int &c = z;
  constexpr int d = a;

  // FIXME: Provide FixIts for static data members too.
#if 0
  struct S {
    static constexpr int b; // xpected-error {{requires an initializer}}
    // -> const int b;
  };

  constexpr int S::b = 0;
#endif

  struct S {
    static char *const p = 0; // expected-error {{requires 'constexpr' specifier}}
    // -> constexpr static char *const p = 0;
  };
}
