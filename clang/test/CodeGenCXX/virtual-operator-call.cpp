// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

struct A {
  virtual int operator-() = 0;
};

void f(A *a) {
  // CHECK: call i32 %
  -*a;
}
