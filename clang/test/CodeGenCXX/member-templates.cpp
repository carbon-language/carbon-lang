// RUN: clang-cc %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// CHECK: ; ModuleID
struct A {
  template<typename T>
  A(T);
};

template<typename T> A::A(T) {}
