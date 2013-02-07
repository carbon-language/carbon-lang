// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -analyze -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);

void use(int);
id foo(int x) {
  if (x)
    return 0;
  static id p = foo(1); 
    clang_analyzer_eval(p == 0); // expected-warning{{TRUE}}
  return p;
}