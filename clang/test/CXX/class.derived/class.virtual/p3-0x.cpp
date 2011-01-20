// RUN: %clang_cc1 -fsyntax-only -std=c++0x -verify %s 

namespace Test1 {

struct B {
  virtual void f(int);
};

struct D : B {
  virtual void f(long) override; // expected-error {{'f' marked 'override' but does not override any member functions}}
  void f(int) override;
};
}

namespace Test2 {

struct A {
  virtual void f(int, char, int);
};

template<typename T>
struct B : A {
  virtual void f(T) override;
};

}
