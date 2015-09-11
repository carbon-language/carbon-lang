// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fno-inline -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: define void @i_want_bar()
// CHECK-NOT: foo
// CHECK: ret void

void bar() {
}

inline void __attribute__((__always_inline__)) foo() {
  bar();
}

void i_want_bar() {
  foo();
}
