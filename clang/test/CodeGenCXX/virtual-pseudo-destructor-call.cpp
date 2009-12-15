// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

struct A {
  virtual ~A();
};

void f(A *a) {
  // CHECK: call void %
  a->~A();
}
