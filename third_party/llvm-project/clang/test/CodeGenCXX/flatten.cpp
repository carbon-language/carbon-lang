// RUN: %clang_cc1 -triple=x86_64-linux-gnu -std=c++11 %s -emit-llvm -o - | FileCheck %s

void f(void) {}

[[gnu::flatten]]
// CHECK: define{{.*}} void @_Z1gv()
void g(void) {
  // CHECK-NOT: call {{.*}} @_Z1fv
  f();
}
