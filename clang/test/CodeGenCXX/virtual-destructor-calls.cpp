// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin10 | FileCheck %s

struct A {
  virtual ~A();
};

struct B : A {
  virtual ~B();
};

// Complete dtor.
// CHECK: define void @_ZN1BD1Ev
// CHECK: call void @_ZN1AD2Ev

// Deleting dtor.
// CHECK: define void @_ZN1BD0Ev
// CHECK: call void @_ZN1AD2Ev
// check: call void @_ZdlPv

// Base dtor.
// CHECK: define void @_ZN1BD2Ev
// CHECK: call void @_ZN1AD2Ev

B::~B() { }
