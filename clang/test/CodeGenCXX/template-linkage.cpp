// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s
template<typename T> struct A {
  virtual void f(T) { }
  inline void g() { } 
};

// Explicit instantiations have external linkage.

// CHECK: define void @_ZN1AIiE1gEv(
template void A<int>::g();

// CHECK: define void @_ZN1AIfE1fEf(
// CHECK: define void @_ZN1AIfE1gEv(
// FIXME: This should also emit the vtable.
template struct A<float>;

// CHECK: define void @_Z1fIiEvT_
template <typename T> void f(T) { }
template void f<int>(int);

// CHECK: define void @_Z1gIiEvT_
template <typename T> inline void g(T) { }
template void g<int>(int);

