// RUN: clang %s -verify -fsyntax-only

int a;
struct {int x;} x = a; // expected-error {{incompatible type initializing 'int', expected 'struct <anonymous>'}}
