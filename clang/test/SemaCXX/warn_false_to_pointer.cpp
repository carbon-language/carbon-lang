// RUN: %clang_cc1 -fsyntax-only -verify %s

int* j = false; // expected-warning{{ initialization of pointer of type 'int *' from literal 'false'}}

void foo(int* i, int *j=(false)) // expected-warning{{ initialization of pointer of type 'int *' from literal 'false'}}
{
  foo(false); // expected-warning{{ initialization of pointer of type 'int *' from literal 'false'}}
  foo((int*)false);
}

