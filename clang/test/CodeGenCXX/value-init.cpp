// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

struct A {
  virtual ~A();
};

struct B : A { };

struct C {
  int i;
  B b;
};

// CHECK: _Z15test_value_initv
void test_value_init() {
  // This value initialization requires zero initialization of the 'B'
  // subobject followed by a call to its constructor.
  // PR5800

  // CHECK: store i32 17
  // CHECK: call void @llvm.memset.i64
  // CHECK: call void @_ZN1BC1Ev
  C c = { 17 } ;
  // CHECK: call void @_ZN1CD1Ev
}
