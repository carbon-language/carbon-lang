// RUN: %clang_cc1 -fsyntax-only -verify %s

int* j = false; // expected-warning{{ initialization of pointer of type 'int *' from literal 'false'}}

void foo(int* i, int *j=(false)) // expected-warning{{ initialization of pointer of type 'int *' from literal 'false'}}
{
  foo(false); // expected-warning{{ initialization of pointer of type 'int *' from literal 'false'}}
  foo((int*)false);
}

char f(struct Undefined*);
double f(...);

// Ensure that when using false in metaprogramming machinery its conversion
// isn't flagged.
template <int N> struct S {};
S<sizeof(f(false))> s;
