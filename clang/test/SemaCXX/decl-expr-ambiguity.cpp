// RUN: %clang_cc1 -fsyntax-only -verify -pedantic-errors %s 

void f() {
  int a;
  struct S { int m; };
  typedef S *T;

  // Expressions.
  T(a)->m = 7;
  int(a)++; // expected-error {{expression is not assignable}}
  __extension__ int(a)++; // expected-error {{expression is not assignable}}
  __typeof(int)(a,5)<<a; // expected-error {{function-style cast to a builtin type can only take one argument}}
  void(a), ++a; // expected-warning {{expression result unused}}
  if (int(a)+1) {}
  for (int(a)+1;;) {} // expected-warning {{expression result unused}}
  a = sizeof(int()+1);
  a = sizeof(int(1));
  typeof(int()+1) a2; // expected-error {{extension used}}
  (int(1)); // expected-warning {{expression result unused}}

  // type-id
  (int())1; // expected-error {{C-style cast from 'int' to 'int ()' is not allowed}}

  // Declarations.
  int fd(T(a)); // expected-warning {{parentheses were disambiguated as a function declarator}}
  T(*d)(int(p)); // expected-warning {{parentheses were disambiguated as a function declarator}} expected-note {{previous definition is here}}
  T(d)[5]; // expected-error {{redefinition of 'd'}}
  typeof(int[])(f) = { 1, 2 }; // expected-error {{extension used}}
  void(b)(int);
  int(d2) __attribute__(()); 
  if (int(a)=1) {}
  int(d3(int()));
}

class C { };
void fn(int(C)) { } // void fn(int(*fp)(C c)) { } expected-note{{candidate function}}
                    // not: void fn(int C);
int g(C);

void foo() {
  fn(1); // expected-error {{no matching function}}
  fn(g); // OK
}
