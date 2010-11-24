// RUN: %clang_cc1 -analyze -analyzer-check-objc-mem -analyzer-inline-call -analyzer-store region -verify %s

struct A {
  int x;
  A(int a) { x = a; }
  int getx() const { return x; }
};

void f1() {
  A x(3);
  if (x.getx() == 3) {
    int *p = 0;
    *p = 3;  // expected-warning{{Dereference of null pointer}}
  } else {
    int *p = 0;
    *p = 3;  // no-warning
  }
}

