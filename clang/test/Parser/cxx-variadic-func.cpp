// RUN: %clang_cc1 -fsyntax-only -verify %s

void f(...) {
  int g(int(...)); // no warning, unambiguously a function declaration
}

void h(int n..., int m); // expected-error {{expected ')'}} expected-note {{to match}}
