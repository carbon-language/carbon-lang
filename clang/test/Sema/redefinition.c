// RUN: clang %s -fsyntax-only -verify
int f(int a) { } // expected-note {{previous definition is here}}
int f(int);
int f(int a) { } // expected-error {{redefinition of 'f'}}

