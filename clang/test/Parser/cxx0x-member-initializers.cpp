// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

// Make sure we don't run off the end of the stream when parsing a deferred
// initializer.
int a; // expected-note {{previous}}
struct S {
  int n = 4 + ; // expected-error {{expected expression}}
} a; // expected-error {{redefinition}}

// Make sure we use all of the tokens.
struct T {
  int a = 1 // expected-error {{expected ';' at end of declaration list}}
  int b = 2;
  int c = b; // expected-error {{undeclared identifier}}
};
