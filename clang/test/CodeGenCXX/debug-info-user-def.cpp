// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin -std=c++11 %s -o - | FileCheck %s

class A {
};

template <typename T> class B {
  T t;
};

A a;
B<int> b;

// Check that no subprograms are emitted into debug info.
// CHECK-NOT: [ DW_TAG_subprogram ]
