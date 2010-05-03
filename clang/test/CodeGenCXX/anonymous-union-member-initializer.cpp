// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s

struct A {
  union {
    int a;
    void* b;
  };
  
  A() : a(0) { }
};

A a;

namespace PR7021 {
  struct X
  {
    union { long l; };
  };

  // CHECK: define void @_ZN6PR70211fENS_1XES0_
  void f(X x, X z) {
    X x1;

    // CHECK: store i64 1, i64
    x1.l = 1;

    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
    X x2(x1);

    X x3;
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
    x3 = x1;

    // CHECK: ret void
  }
}
