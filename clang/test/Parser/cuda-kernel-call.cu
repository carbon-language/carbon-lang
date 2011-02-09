// RUN: %clang_cc1 -fsyntax-only -verify %s

void foo(void) {
  foo<<<1;      // expected-error {{expected '>>>'}} expected-note {{to match this '<<<'}}

  foo<<<1,1>>>; // expected-error {{expected '('}}

  foo<<<>>>();  // expected-error {{expected expression}}
}
