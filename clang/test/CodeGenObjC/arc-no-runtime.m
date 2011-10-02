// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-arc -emit-llvm %s -o - | FileCheck %s

// rdar://problem/9224855
void test0() {
  id x = 0;
  // CHECK: call void @objc_release(
}

// CHECK: declare extern_weak void @objc_release(
