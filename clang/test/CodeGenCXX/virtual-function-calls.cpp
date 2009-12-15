// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

// PR5021
struct A {
  virtual void f(char);
};

void f(A *a) {
  // CHECK: call void %
  a->f('c');
}
