// RUN: clang %s -verify -fsyntax-only

void foo(void);
void foo(void) {} // expected-error{{previous definition is here}}
void foo(void);
void foo(void);

void foo(int); // expected-error {{redefinition of 'foo'}}
