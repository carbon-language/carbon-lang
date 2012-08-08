// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -O0 -fms-extensions -w -emit-llvm -o - | FileCheck %s

void t1() {
// CHECK: @t1
// CHECK: ret void
  __asm {}
}
