// RUN: %clang_cc1 -fms-extensions -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck %s

class A {
 public:
  ~A() {}
};

int f() {
// CHECK: @_Z1fv
// CHECK-NOT: call void @_ZN1AD1Ev
// CHECK: ret i32 0
  return __noop(A());
};
