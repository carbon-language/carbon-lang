// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fexceptions -o - %s | FileCheck %s

struct X0 {
  X0();
  X0(const X0 &) throw();
  X0(X0 &);
};

struct X1 {
  X1();
  X1(const X1 &) throw();
};

struct X2 : X1 { 
  X2();
};
struct X3 : X0, X1 { 
  X3();
};

struct X4 {
  X4(X4 &) throw();
};

struct X5 : X0, X4 { };

void test(X2 x2, X3 x3, X5 x5) {
  // CHECK: define linkonce_odr void @_ZN2X2C1ERKS_
  // CHECK-NOT: define
  // CHECK: call void @__cxa_call_unexpected
  // CHECK-NOT: define
  // CHECK: ret void
  X2 x2a(x2);
  // CHECK: define linkonce_odr void @_ZN2X3C1ERKS_
  // CHECK-NOT: define
  // CHECK: call void @__cxa_call_unexpected
  // CHECK-NOT: define
  // CHECK: ret void
  X3 x3a(x3);
  // CHECK: define linkonce_odr void @_ZN2X5C1ERS_
  // CHECK-NOT: call void @__cxa_call_unexpected
  // CHECK: ret void
  X5 x5a(x5);
}
