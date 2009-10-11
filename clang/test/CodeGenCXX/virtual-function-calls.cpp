// RUN: clang-cc %s -emit-llvm -o - | FileCheck %s

// PR5021
struct A {
  virtual void f(char);
};

void f(A *a) {
  a->f('c');
}

void f(A a) {
  // This should not be a virtual function call.
  
  // CHECK: call void @_ZN1A1fEc
  a.f('c');
}