// RUN: %clang_cc1 %s -cxx-abi itanium -emit-llvm-only

struct A { virtual ~A(); };
struct B : A {
  ~B() { }
};
B x;

