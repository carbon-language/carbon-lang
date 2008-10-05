// RUN: clang -fsyntax-only -verify -pedantic-errors %s 

void f() {
  int a;
  struct S { int m; };
  typedef S *T;

  // Expressions.
  T(a)->m = 7;
  int(a)++; // expected-error {{invalid lvalue in increment/decrement expression}}
  __extension__ int(a)++; // expected-error {{invalid lvalue in increment/decrement expression}}
  typeof(int)(a,5)<<a; // expected-error {{function-style cast to a builtin type can only take one argument}}
  void(a), ++a; // expected-warning {{statement was disambiguated as expression}} expected-warning {{expression result unused}}
  if (int(a)+1) {}
  for (int(a)+1;;) {}
  a = sizeof(int()+1);
  a = sizeof(int(1));
  typeof(int()+1) a2;
  (int(1)); // expected-warning {{expression result unused}}

  // type-id
  (int())1; // expected-error {{used type 'int ()' where arithmetic or pointer type is required}}

  // Declarations.
  T(*d)(int(p)); // expected-warning {{statement was disambiguated as declaration}} expected-error {{previous definition is here}}
  T(d)[5]; // expected-warning {{statement was disambiguated as declaration}} expected-error {{redefinition of 'd'}}
  typeof(int[])(f) = { 1, 2 }; // expected-warning {{statement was disambiguated as declaration}}
  void(b)(int);
  int(d2) __attribute__(()); // expected-warning {{statement was disambiguated as declaration}}
  if (int(a)=1) {}
  int(d3(int())); // expected-warning {{statement was disambiguated as declaration}}
}

class C { };
void fn(int(C)) { } // void fn(int(*fp)(C c)) { }
                    // not: void fn(int C);
int g(C);

void foo() {
  fn(1); // expected-error {{incompatible integer to pointer conversion passing 'int', expected 'int (*)(class C)'}}
  fn(g); // OK
}
