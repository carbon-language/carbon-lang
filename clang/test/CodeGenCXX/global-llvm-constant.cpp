// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

struct A {
  A() { x = 10; }
  int x;
};

const A x;

// CHECK: @_ZL1x = internal global
