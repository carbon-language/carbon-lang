// RUN: %clang_cc1 %s -emit-llvm-only
// PR6294

class A {
  virtual ~A();
};
class B {
  class C;
};
class B::C : public A {
  C();
};
B::C::C() {}
