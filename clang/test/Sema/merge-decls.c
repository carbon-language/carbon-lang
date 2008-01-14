// RUN: clang %s -verify -fsyntax-only

void foo(void);
void foo(void) {} 
void foo(void);
void foo(void); // expected-error{{previous definition is here}}

void foo(int); // expected-error {{redefinition of 'foo'}}

int funcdef()
{
 return 0;
}

int funcdef();

int funcdef2() { return 0; } // expected-error{{previous definition is here}}
int funcdef2() { return 0; } // expected-error {{redefinition of 'funcdef2'}}

