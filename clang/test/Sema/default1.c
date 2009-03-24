// RUN: clang-cc -fsyntax-only -verify %s
void f(int i = 0); // expected-error {{C does not support default arguments}}
