// RUN: %clang_cc1 %s -fno-rtti -cxx-abi microsoft -triple=i386-pc-win32 -emit-llvm -o %t

struct A {};
struct B : virtual A {
  virtual ~B();
};
struct C : B {
  C();
};

C::C() {}
