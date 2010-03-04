// RUN: %clang_cc1 -triple arm-unknown-unknown -emit-llvm -o - %s | FileCheck %s

// XFAIL: win32

// CHECK: declare arm_aapcscc i32 @printf(i8*, ...)
void f0() {
  printf("a\n");
}

// CHECK: call arm_aapcscc void @exit
// CHECK: unreachable
void f1() {
  exit(1);
}
