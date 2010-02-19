// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin10 -mconstructor-aliases | FileCheck %s

struct A {
  virtual ~A();
};

struct B : A {
  virtual ~B();
};

// Complete dtor: just an alias because there are no virtual bases.
// CHECK: @_ZN1BD1Ev = alias {{.*}} @_ZN1BD2Ev

// Deleting dtor: defers to the complete dtor.
// CHECK: define void @_ZN1BD0Ev
// CHECK: call void @_ZN1BD1Ev
// CHECK: call void @_ZdlPv

// Base dtor: actually calls A's base dtor.
// CHECK: define void @_ZN1BD2Ev
// CHECK: call void @_ZN1AD2Ev

B::~B() { }
