// RUN: %clang_cc1 -fsyntax-only -verify %s

- im0 { int a; return 0; // expected-error{{missing context for method declaration}}
// expected-error{{expected '}'}}
