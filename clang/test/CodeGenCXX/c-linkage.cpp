// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
// pr6644

extern "C" {
  namespace N {
    struct X { 
      virtual void f();
    };
    void X::f() { }
  }
}

// CHECK: define void @_ZN1N1X1fEv

extern "C" {
  static void test2_f() {
  }
  // This is not required by the standard, but users assume they know
  // the mangling of static functions in extern "C" contexts.
  // CHECK: define internal void @test2_f(
  void test2_use() {
    test2_f();
  }
}

extern "C" {
  struct test3_s {
  };
  bool operator==(const int& a, const test3_s& b)  {
  }
}
