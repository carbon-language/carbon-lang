// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s

// PR7180
int f(a::b::c); // expected-error {{use of undeclared identifier 'a'}}

class Foo::Bar { // expected-error {{use of undeclared identifier 'Foo'}} \
                 // expected-note {{to match this '{'}} \
                 // expected-error {{expected ';' after class}}
                 // expected-error {{expected '}'}}
