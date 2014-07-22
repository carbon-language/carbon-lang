// RUN: %clang_cc1 -std=gnu++11 -fsyntax-only -fms-compatibility -verify %s

void f() {
  // GNU-style attributes are prohibited in this position.
  auto P = new int * __attribute__((vector_size(8))); // expected-error {{an attribute list cannot appear here}} \
                                                      // expected-error {{invalid vector element type 'int *'}}

  // Ensure that MS type attribute keywords are still supported in this
  // position.
  auto P2 = new int * __sptr; // Ok
}

void g(int a[static [[]] 5]); // expected-error {{static array size is a C99 feature, not permitted in C++}}
