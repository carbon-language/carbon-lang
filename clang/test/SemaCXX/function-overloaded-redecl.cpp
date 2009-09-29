// RUN: clang-cc -fsyntax-only -verify %s 

typedef const int cInt;

void f (int); 
void f (const int);  // redecl

void f (int) {  }    // expected-note {{previous definition is here}}
void f (cInt) { }    // expected-error {{redefinition of 'f'}}

