// RUN: %clang_cc1 -fsyntax-only -std=c++0x -verify %s 

struct B {
  virtual void f(int);
};

struct D : B {
  virtual void f(long) override; // expected-error {{'f' marked 'override' but does not override any member functions}}
  void f(int) override;
};
