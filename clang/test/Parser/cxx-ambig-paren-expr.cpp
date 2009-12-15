// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s

void f() {
  typedef int T;
  int x, *px;
  
  // Type id.
  (T())x;    // expected-error {{cast from 'int' to 'T ()'}}
  (T())+x;   // expected-error {{cast from 'int' to 'T ()'}}
  (T())*px;  // expected-error {{cast from 'int' to 'T ()'}}
  
  // Expression.
  x = (T());
  x = (T())/x;

  typedef int *PT;
  // Make sure stuff inside the parens are parsed only once (only one warning).
  x = (PT()[(int){1}]); // expected-warning {{compound literals}}

  // Special case: empty parens is a call, not an expression
  struct S{int operator()();};
  (S())();

  // FIXME: Special case: "++" is postfix here, not prefix
  // (S())++;
}
