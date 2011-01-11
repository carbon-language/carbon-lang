// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin10 -mconstructor-aliases | FileCheck %s

struct Member {
  ~Member();
};

struct A {
  virtual ~A();
};

struct B : A {
  Member m;
  virtual ~B();
};

// Complete dtor: just an alias because there are no virtual bases.
// CHECK: @_ZN1BD1Ev = alias {{.*}} @_ZN1BD2Ev

// (aliases from C)
// CHECK: @_ZN1CD1Ev = alias {{.*}} @_ZN1CD2Ev
// CHECK: @_ZN1CD2Ev = alias bitcast {{.*}} @_ZN1BD2Ev

// Deleting dtor: defers to the complete dtor.
// CHECK: define unnamed_addr void @_ZN1BD0Ev
// CHECK: call void @_ZN1BD1Ev
// CHECK: call void @_ZdlPv

// Base dtor: actually calls A's base dtor.
// CHECK: define unnamed_addr void @_ZN1BD2Ev
// CHECK: call void @_ZN6MemberD1Ev
// CHECK: call void @_ZN1AD2Ev

B::~B() { }

struct C : B {
  ~C();
};

C::~C() { }

// Complete dtor: just an alias (checked above).

// Deleting dtor: defers to the complete dtor.
// CHECK: define unnamed_addr void @_ZN1CD0Ev
// CHECK: call void @_ZN1CD1Ev
// CHECK: call void @_ZdlPv

// Base dtor: just an alias to B's base dtor.
