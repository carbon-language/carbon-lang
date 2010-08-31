// // RUN: %clang_cc1 -analyze -analyzer-check-objc-mem -analyzer-inline-call -analyzer-store region -verify %s
 
// FIXME: Super-simple test to make sure we don't die on temporaries.

struct X {
  X();
  ~X();
  X operator++(int);
};

int f(X x, X y) {
  for (; ; x++) { }
}
