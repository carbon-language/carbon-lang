// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

// PR12219
struct A { A(int); virtual ~A(); };
struct B : A { using A::A; ~B(); };
B::~B() {}

B b(123);

struct C { template<typename T> C(T); };
struct D : C { using C::C; };
D d(123);

// CHECK: define void @_ZN1BD0Ev
// CHECK: define void @_ZN1BD1Ev
// CHECK: define void @_ZN1BD2Ev

// CHECK: define linkonce_odr void @_ZN1BC1Ei(
// CHECK: call void @_ZN1BC2Ei(

// CHECK: define linkonce_odr void @_ZN1DC1IiEET_(
// CHECK: call void @_ZN1DC2IiEET_(

// CHECK: define linkonce_odr void @_ZN1DC2IiEET_(
// CHECK: call void @_ZN1CC2IiEET_(

// CHECK: define linkonce_odr void @_ZN1BC2Ei(
// CHECK: call void @_ZN1AC2Ei(
