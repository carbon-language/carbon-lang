// RUN: clang-cc -fsyntax-only %s -verify

void a() {goto A; // expected-error {{use of undeclared label}}
// expected-error {{expected '}'}}
