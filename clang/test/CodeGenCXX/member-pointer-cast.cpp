// RUN: clang-cc %s -emit-llvm -o - -triple=x86_64-apple-darwin9 | FileCheck %s

struct A { int a; };
struct B { int b; };
struct C : B, A { };

int A::*pa;
int C::*pc;

void f() {
  // CHECK: store i64 -1, i64* @pa
  pa = 0;

  // CHECK: [[ADJ:%[a-zA-Z0-9\.]+]] = add i64 {{.*}}, 4
  // CHECK: store i64 [[ADJ]], i64* @pc
  pc = pa;

  // CHECK: [[ADJ:%[a-zA-Z0-9\.]+]] = sub i64 {{.*}}, 4
  // CHECK: store i64 [[ADJ]], i64* @pa
  pa = static_cast<int A::*>(pc);
}
