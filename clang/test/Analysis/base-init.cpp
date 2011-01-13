// RUN: %clang_cc1 -analyze -analyzer-check-objc-mem -analyzer-store region -analyzer-inline-call -cfg-add-initializers -verify %s

class A {
  int x;
public:
  A();
  int getx() const {
    return x;
  }
};

A::A() : x(0) {
}

class B : public A {
  int y;
public:
  B();
};

B::B() {
}

void f() {
  B b;
  if (b.getx() != 0) {
    int *p = 0;
    *p = 0; // no-warning
  }
}
