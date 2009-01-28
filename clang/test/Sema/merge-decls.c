// RUN: clang %s -verify -fsyntax-only

void foo(void);
void foo(void) {} 
void foo(void);
void foo(void); // expected-note {{previous declaration is here}}

void foo(int); // expected-error {{conflicting types for 'foo'}}

int funcdef()
{
 return 0;
}

int funcdef();

int funcdef2() { return 0; } // expected-note {{previous definition is here}}
int funcdef2() { return 0; } // expected-error {{redefinition of 'funcdef2'}}

// PR2502
void (*f)(void);
void (*f)() = 0;
