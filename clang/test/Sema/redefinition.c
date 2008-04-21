// RUN: clang %s -fsyntax-only -verify
int f(int) { } // expected-error{{previous definition is here}}
int f(int);
int f(int) { } // expected-error{{redefinition of 'f'}}

