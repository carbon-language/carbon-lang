// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -analyze -analyzer-checker=core,debug.ExprInspection %s -std=c++11 -verify

void clang_analyzer_eval(bool);

void test(int x) {
  clang_analyzer_eval(&x == __builtin_addressof(x)); // expected-warning{{TRUE}}
}
