// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fno-inline -emit-llvm %s -o - | FileCheck %s

// CHECK-NOT: foo

void bar() {
}

inline void __attribute__((__always_inline__)) foo() {
  bar();
}

void i_want_bar() {
  foo();
}
