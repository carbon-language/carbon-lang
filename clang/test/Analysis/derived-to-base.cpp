// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-store region %s

class A {
protected:
  int x;
};

class B : public A {
public:
  void f();
};

void B::f() {
  x = 3;
}


class C : public B {
public:
  void g() {
    // This used to crash because we are upcasting through two bases.
    x = 5;
  }
};
