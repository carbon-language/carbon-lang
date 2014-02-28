// RUN: %clang_cc1 -analyzer-config graph-trim-interval=1 -analyzer-max-loop 1 -analyze -analyzer-checker=core,alpha.cplusplus.NewDeleteLeaks -verify %s
// expected-no-diagnostics

class A {
public:
  void f() {};
  ~A() {
    for (int i=0; i<3; i++)
      f();
  }
};

void error() {
  A *a = new A();
  delete a;
}
