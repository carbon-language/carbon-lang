// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s

void f(void);
void g(void) {
  // CHECK: define void @g() #0
  // CHECK-NOT: call void @f() nounwind
  f();
}

// CHECK-NOT: declare void @f() #0

// CHECK: attributes #0 = { nounwind{{.*}} }
