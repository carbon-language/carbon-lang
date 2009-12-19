// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

// PR5021
struct A {
  virtual void f(char);
};

void f(A *a) {
  // CHECK: call void %
  a->f('c');
}

struct B : virtual A { 
  virtual void f();
};

void f(B * b) {
  b->f();
}