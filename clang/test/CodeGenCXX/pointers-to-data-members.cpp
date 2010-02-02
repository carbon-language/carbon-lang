// RUN: %clang_cc1 %s -emit-llvm -o - -triple=x86_64-apple-darwin10 | FileCheck %s

struct A { int a; };
struct B { int b; };
struct C : B, A { };

// Casts.
namespace Casts {

int A::*pa;
int C::*pc;

void f() {
  // CHECK: store i64 -1, i64* @_ZN5Casts2paE
  pa = 0;

  // CHECK: [[ADJ:%[a-zA-Z0-9\.]+]] = add i64 {{.*}}, 4
  // CHECK: store i64 [[ADJ]], i64* @_ZN5Casts2pcE
  pc = pa;

  // CHECK: [[ADJ:%[a-zA-Z0-9\.]+]] = sub i64 {{.*}}, 4
  // CHECK: store i64 [[ADJ]], i64* @_ZN5Casts2paE
  pa = static_cast<int A::*>(pc);
}

}
