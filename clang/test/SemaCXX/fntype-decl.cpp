// RUN: clang -fsyntax-only -verify %s

// PR2942
typedef void fn(int);
fn f; // expected-error{{previous declaration is here}}

int g(int x, int y);
int g(int x, int y = 2);

typedef int g_type(int, int);
g_type g;

int h(int x) { // expected-error{{previous definition is here}}
  return g(x);
}

float f(int) { } // expected-error{{functions that differ only in their return type cannot be overloaded}}

int h(int) { } // expected-error{{redefinition of 'h'}}

