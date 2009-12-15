// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

namespace test1 {
int x;
template <int& D> class T { };
// CHECK: void @_ZN5test12f0ENS_1TILZNS_1xEEEE(
void f0(T<x> a0) {}
}

namespace test1 {
// CHECK: void @_ZN5test12f0Ef
void f0(float) {}
template<void (&)(float)> struct t1 {};
// CHECK: void @_ZN5test12f1ENS_2t1ILZNS_2f0EfEEE(
void f1(t1<f0> a0) {}
}

namespace test2 {
// CHECK: void @_ZN5test22f0Ef
void f0(float) {}
template<void (*)(float)> struct t1 {};
// FIXME: Fails because we don't treat as an expression.
// CHECK-FIXME: void @_ZN5test22f1ENS_2t1IXadL_ZNS_2f0EfEEEE(
void f1(t1<f0> a0) {}
}

namespace test3 {
// CHECK: void @test3_f0
extern "C" void test3_f0(float) {}
template<void (&)(float)> struct t1 {};
// FIXME: Fails because we tack on a namespace.
// CHECK-FIXME: void @_ZN5test32f1ENS_2t1ILZ8test3_f0EEE(
void f1(t1<test3_f0> a0) {}
}

namespace test4 {
// CHECK: void @test4_f0
extern "C" void test4_f0(float) {}
template<void (*)(float)> struct t1 {};
// FIXME: Fails because we don't treat as an expression.
// CHECK-FIXME: void @_ZN5test42f1ENS_2t1IXadL_Z8test4_f0EEEE(
void f1(t1<test4_f0> a0) {}
}

// CHECK: void @test5_f0
extern "C" void test5_f0(float) {}
int main(int) {}

namespace test5 {
template<void (&)(float)> struct t1 {};
// CHECK: void @_ZN5test52f1ENS_2t1ILZ8test5_f0EEE(
void f1(t1<test5_f0> a0) {}

template<int (&)(int)> struct t2 {};
// CHECK: void @_ZN5test52f2ENS_2t2ILZ4mainEEE
void f2(t2<main> a0) {}
}

// FIXME: This fails.
namespace test6 {
struct A { void im0(float); };
// CHECK: void @_ZN5test61A3im0Ef
void A::im0(float) {}
template <void(A::*)(float)> class T { };
// FIXME: Fails because we don't treat as an expression.
// CHECK-FAIL: void @_ZN5test62f0ENS_1TIXadL_ZNS_1A3im0EfEEEE(
void f0(T<&A::im0> a0) {}
}
