// RUN: clang-cc %s -verify -fsyntax-only

int a;
struct {int x;} x = a; // expected-error {{incompatible type initializing 'int', expected 'struct <anonymous>'}}
