// RUN: clang %s -verify -fsyntax-only
typedef char T[4];

T foo(int n, int m) {  }  // expected-error {{cannot return array or function}}

