// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

namespace Test1 {
  struct A {
    virtual int f() final;
  };

  // CHECK: define i32 @_ZN5Test11fEPNS_1AE
  int f(A *a) {
    // CHECK: call i32 @_ZN5Test11A1fEv
    return a->f();
  }
}

namespace Test2 {
  struct A final {
    virtual int f();
  };

  // CHECK: define i32 @_ZN5Test21fEPNS_1AE
  int f(A *a) {
    // CHECK: call i32 @_ZN5Test21A1fEv
    return a->f();
  }
}
