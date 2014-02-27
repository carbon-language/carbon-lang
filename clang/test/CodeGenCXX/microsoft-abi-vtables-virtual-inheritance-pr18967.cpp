// RUN: not %clang_cc1 -triple i686-pc-win32 -emit-llvm-only -fno-rtti %s 2>&1 | FileCheck %s

// CHECK: error: v-table layout for classes with non-virtual base classes that override methods in virtual bases is not supported yet

struct A {
  virtual int foo() { return a; }
  int a;
};
struct B : virtual A {
  B() : b(1) {}
  virtual int bar() { return b; }
  int b;
};
struct C : virtual A {
  C() : c(2) {}
  virtual int foo() { return c; }
  int c;
};
struct D : B, C {
  D() : d(3) {}
  virtual int bar() { return d; }
  int d;
};
int main() {
  D d;
  return d.foo();
}
