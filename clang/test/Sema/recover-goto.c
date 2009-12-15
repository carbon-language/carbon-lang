// RUN: %clang_cc1 -fsyntax-only %s -verify

void a() {goto A; // expected-error {{use of undeclared label}}
// expected-error {{expected '}'}}
