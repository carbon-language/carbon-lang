// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin10 -mconstructor-aliases | FileCheck %s

struct A { 
  A();
};

// CHECK: @_ZN1AC1Ev = alias {{.*}} @_ZN1AC2Ev
// CHECK: define unnamed_addr void @_ZN1AC2Ev(%struct.A* %this)
A::A() { }

struct B : virtual A { 
  B();
};

// CHECK: define unnamed_addr void @_ZN1BC1Ev(%struct.B* %this)
// CHECK: define unnamed_addr void @_ZN1BC2Ev(%struct.B* %this, i8** %vtt)
B::B() { }

struct C : virtual A {
  C(bool);
};

// CHECK: define unnamed_addr void @_ZN1CC1Eb(%struct.B* %this, i1 zeroext)
// CHECK: define unnamed_addr void @_ZN1CC2Eb(%struct.B* %this, i8** %vtt, i1 zeroext)
C::C(bool) { }

// PR6251
namespace PR6251 {

// Test that we don't call the A<char> constructor twice.

template<typename T>
struct A { A(); };

struct B : virtual A<char> { };
struct C : virtual A<char> { };

struct D : B, C  {
  D();
};

// CHECK: define unnamed_addr void @_ZN6PR62511DC1Ev
// CHECK: call void @_ZN6PR62511AIcEC2Ev
// CHECK-NOT: call void @_ZN6PR62511AIcEC2Ev
// CHECK: ret void
D::D() { }

}
