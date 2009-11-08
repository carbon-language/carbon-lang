// RUN: clang-cc -fsyntax-only -verify %s 
void f(int) { } // expected-note {{previous definition is here}}
void f(const int) { } // expected-error {{redefinition of 'f'}}
