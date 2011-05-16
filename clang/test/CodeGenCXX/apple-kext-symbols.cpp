// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fapple-kext -emit-llvm -o - %s | FileCheck %s

// rdar://problem/9429976
namespace test0 {
  struct A {
    A();
    virtual ~A();
    virtual void foo() = 0;
  };

  // CHECK: define void @_ZN5test01AC1Ev(
  // CHECK: define void @_ZN5test01AC2Ev(
  A::A() {}

  // CHECK: define void @_ZN5test01AD0Ev(
  // CHECK: define void @_ZN5test01AD1Ev(
  // CHECK: define void @_ZN5test01AD2Ev(
  A::~A() {}
}

