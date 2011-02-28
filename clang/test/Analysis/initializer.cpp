// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-store region -cfg-add-initializers -verify %s

class A {
  int x;
public:
  A();
};

A::A() : x(0) {
  if (x != 0) {
    int *p = 0;
    *p = 0; // no-warning
  }
}
