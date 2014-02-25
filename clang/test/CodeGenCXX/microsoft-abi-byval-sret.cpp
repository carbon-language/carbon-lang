// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i686-pc-win32 -mconstructor-aliases -fno-rtti | FileCheck %s

struct A {
  A() : a(42) {}
  A(const A &o) : a(o.a) {}
  ~A() {}
  int a;
  A foo(A o);
};

A A::foo(A x) {
  A y(*this);
  y.a += x.a;
  return y;
}

// CHECK-LABEL: define x86_thiscallcc %struct.A* @"\01?foo@A@@QAE?AU1@U1@@Z"
// CHECK:       (%struct.A* %this, <{ %struct.A*, %struct.A }>* inalloca)
// CHECK:   getelementptr inbounds <{ %struct.A*, %struct.A }>* %{{.*}}, i32 0, i32 0
// CHECK:   load %struct.A**
// CHECK:   ret %struct.A*

int main() {
  A x;
  A y = x.foo(x);
}

// CHECK: call x86_thiscallcc %struct.A* @"\01?foo@A@@QAE?AU1@U1@@Z"
// CHECK:       (%struct.A* %{{[^,]*}}, <{ %struct.A*, %struct.A }>* inalloca %{{[^,]*}})
