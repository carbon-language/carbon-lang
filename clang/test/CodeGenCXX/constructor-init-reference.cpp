// RUN: %clang_cc1 -cxx-abi itanium -emit-llvm -o - %s | FileCheck %s

int x;
struct A {
  int& y;
  A() : y(x) {}
};
A z;
// CHECK: store i32* @x, i32**
