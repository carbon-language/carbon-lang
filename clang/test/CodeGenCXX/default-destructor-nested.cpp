// RUN: %clang_cc1 %s -cxx-abi itanium -emit-llvm-only
// PR6294

class A {
public: virtual ~A();
};
class B {
  class C;
};
class B::C : public A {
  C();
};
B::C::C() {}
