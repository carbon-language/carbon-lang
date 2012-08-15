// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-ipa=dynamic-bifurcate -verify %s

void clang_analyzer_eval(bool);

class A {
public:
  virtual int get() { return 0; }
};

void testBifurcation(A *a) {
  clang_analyzer_eval(a->get() == 0); // expected-warning{{TRUE}} expected-warning{{UNKNOWN}}
}

void testKnown() {
  A a;
  clang_analyzer_eval(a.get() == 0); // expected-warning{{TRUE}}
}
