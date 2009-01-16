// RUN: clang -fsyntax-only -verify %s
/* Test attempt to redefine 'id' in an incompatible fashion.  */

typedef int id;   // expected-error {{typedef redefinition with different types}}

id b;
