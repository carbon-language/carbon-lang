// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

int x() { return 1; }

// CHECK:  [[retval:%.*]] = alloca i32
// CHECK:  store i32 1, i32* [[retval]]
// CHECK:  [[load:%.*]] = load i32* [[retval]]
// CHECK:  ret i32 [[load]]


int f() __attribute__((weak, alias("x")));

/* Test that we link to the alias correctly instead of making a new
   forward definition. */
int f();
int h() {
  return f();
}

// CHECK:  [[retval:%.*]] = alloca i32
// CHECK:  [[call:%.*]] = call i32 (...)* @f()
// CHECK:  store i32 [[call]], i32* [[retval]]
// CHECK:  [[load:%.*]] = load i32* [[retval]]
// CHECK:  ret i32 [[load]]

