// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-ipa=inlining -analyzer-store region -verify %s
// XFAIL: *

void clang_analyzer_eval(bool);

struct A {
  int x;
  A(int a) { x = a; }
  int getx() const { return x; }
};

void f1() {
  A x(3);
  clang_analyzer_eval(x.getx() == 3); // expected-warning{{TRUE}}
}

void f2() {
  const A &x = A(3);
  clang_analyzer_eval(x.getx() == 3); // expected-warning{{TRUE}}
}

void f3() {
  const A &x = (A)3;
  clang_analyzer_eval(x.getx() == 3); // expected-warning{{TRUE}}
}
