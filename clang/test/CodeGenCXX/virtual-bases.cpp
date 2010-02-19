// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin10 -mconstructor-aliases | FileCheck %s

struct A { 
  A();
};

// CHECK: @_ZN1AC1Ev = alias {{.*}} @_ZN1AC2Ev
// CHECK: define void @_ZN1AC2Ev(%struct.A* %this)
A::A() { }

struct B : virtual A { 
  B();
};

// CHECK: define void @_ZN1BC1Ev(%struct.B* %this)
// CHECK: define void @_ZN1BC2Ev(%struct.B* %this, i8** %vtt)
B::B() { }

struct C : virtual A {
  C(bool);
};

// CHECK: define void @_ZN1CC1Eb(%struct.B* %this, i1 zeroext)
// CHECK: define void @_ZN1CC2Eb(%struct.B* %this, i8** %vtt, i1 zeroext)
C::C(bool) { }
