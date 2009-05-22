// RUN: clang-cc -fsyntax-only -verify %s

void f() {
  typedef int T;
  int x, *px;
  
  // Type id.
  (T())x;    // expected-error {{used type 'T (void)'}}
  (T())+x;   // expected-error {{used type 'T (void)'}}
  (T())*px;  // expected-error {{used type 'T (void)'}}
  
  // Expression.
  x = (T());
  x = (T())/x;
}
