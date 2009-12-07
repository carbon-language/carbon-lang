// RUN: clang-cc %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s
// PR5697
namespace PR5697 {
struct A {
  virtual void f() { } 
  A();
  A(int);
};

// A does not have a key function, so the first constructor we emit should
// cause the vtable to be defined (without assertions.)
// CHECK: @_ZTVN6PR56971AE = weak_odr constant
A::A() { }
A::A(int) { }
}
