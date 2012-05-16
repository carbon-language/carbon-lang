// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-store region -cfg-add-initializers -verify %s

void clang_analyzer_eval(bool);

class A {
  int x;
public:
  A();
};

A::A() : x(0) {
  clang_analyzer_eval(x == 0); // expected-warning{{TRUE}}
}
