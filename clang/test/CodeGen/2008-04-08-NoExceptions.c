// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

void f(void);
void g(void) {
  // CHECK: define void @g() nounwind
  // CHECK-NOT: call void @f() nounwind
  f();
}

// CHECK-NOT: declare void @f() nounwind
