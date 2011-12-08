// RUN: %clang_cc1 -x c++ -verify -fsyntax-only %s

void f1();

struct S {
  static void f2();
};

extern void f3() __attribute__((weak_import));

struct S2 {
  static void f4() __attribute__((weak_import));
};

void bar() {
  bool b;

  b = f1; // expected-warning {{address of function 'f1' will always evaluate to 'true'}}
  if (f1) {} // expected-warning {{address of function 'f1' will always evaluate to 'true'}}
  b = S::f2; // expected-warning {{address of function 'S::f2' will always evaluate to 'true'}}
  if (S::f2) {} // expected-warning {{address of function 'S::f2' will always evaluate to 'true'}}

  // implicit casts of weakly imported symbols are ok:
  b = f3;
  if (f3) {}
  b = S2::f4;
  if (S2::f4) {}
}
