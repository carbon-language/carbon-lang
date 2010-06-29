// RUN: %clang_cc1 -triple x86_64 -emit-llvm -o - %s | FileCheck %s

// Check that we don't generate unnecessary reloads.
//
// CHECK: define void @f0()
// CHECK:      [[x_0:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[y_0:%.*]] = alloca i32, align 4
// CHECK-NEXT: store i32 1, i32* [[x_0]]
// CHECK-NEXT: store i32 1, i32* [[x_0]]
// CHECK-NEXT: store i32 1, i32* [[y_0]]
// CHECK: }
void f0() {
  int x, y;
  x = 1;
  y = (x = 1);
}

// Check that we do generate reloads for volatile access.
//
// CHECK: define void @f1()
// CHECK:      [[x_1:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[y_1:%.*]] = alloca i32, align 4
// CHECK-NEXT: volatile store i32 1, i32* [[x_1]]
// CHECK-NEXT: volatile store i32 1, i32* [[x_1]]
// CHECK-NEXT: [[tmp_1:%.*]] = volatile load i32* [[x_1]]
// CHECK-NEXT: volatile store i32 [[tmp_1]], i32* [[y_1]]
// CHECK: }
void f1() {
  volatile int x, y;
  x = 1;
  y = (x = 1);
}
