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
