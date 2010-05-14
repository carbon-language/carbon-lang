// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s

class Foo::Bar { // expected-error {{use of undeclared identifier 'Foo'}} \
                 // expected-note {{to match this '{'}} \
                 // expected-error {{expected ';' after class}}
                 // expected-error {{expected '}'}}
