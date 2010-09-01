// RUN: %clang_cc1 -fsyntax-only -verify %s

- im0 { // expected-note{{to match this '{'}}  expected-error{{missing context for method declaration}}
  int a; return 0;
// expected-error{{expected '}'}}
