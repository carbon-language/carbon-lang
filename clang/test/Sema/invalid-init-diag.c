// RUN: %clang_cc1 %s -verify -fsyntax-only

int a;
struct {int x;} x = a; // expected-error {{incompatible type initializing 'int', expected 'struct <anonymous>'}}
