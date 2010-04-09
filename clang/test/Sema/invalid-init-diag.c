// RUN: %clang_cc1 %s -verify -fsyntax-only

int a;
struct {int x;} x = a; // expected-error {{from an expression of incompatible type 'int'}}
