// RUN: clang-cc -emit-llvm %s -o - -triple=x86_64-apple-darwin9 | FileCheck %s

namespace std {
  struct A { A(); };
  
  // CHECK: define void @_ZNSt1AC1Ev
  // CHECK: define void @_ZNSt1AC2Ev
  A::A() { }
};
