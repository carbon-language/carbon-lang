// RUN: %clang_cc1 %s -O3 -emit-llvm -o - | FileCheck %s

namespace Test1 {
  struct A {
    virtual int f() __attribute__((final)) { return 1; }
  };

  // CHECK: define i32 @_ZN5Test11fEPNS_1AE
  int f(A *a) {
    // CHECK: ret i32 1
    return a->f();
  }
}

namespace Test2 {
  struct __attribute__((final)) A {
    virtual int f() { return 1; }
  };

  // CHECK: define i32 @_ZN5Test21fEPNS_1AE
  int f(A *a) {
    // CHECK: ret i32 1
    return a->f();
  }
}
