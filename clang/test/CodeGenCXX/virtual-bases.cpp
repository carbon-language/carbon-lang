// RUN: clang-cc -emit-llvm %s -o - -triple=x86_64-apple-darwin10 | FileCheck %s

struct A { 
  A();
};

// CHECK: define void @_ZN1AC1Ev(%struct.A* %this)
// CHECK: define void @_ZN1AC2Ev(%struct.A* %this)
A::A() { }

struct B : virtual A { 
  B();
};

// CHECK: define void @_ZN1BC1Ev(%struct.B* %this)
// CHECK: define void @_ZN1BC2Ev(%struct.B* %this, i8** %vtt)
B::B() { }
