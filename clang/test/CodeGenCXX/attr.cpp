// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s

// CHECK: define i32 @_Z3foov() nounwind align 1024
int foo() __attribute__((aligned(1024)));
int foo() { }

class C {
  virtual void bar1() __attribute__((aligned(1)));
  virtual void bar2() __attribute__((aligned(2)));
  virtual void bar3() __attribute__((aligned(1024)));
} c;

// CHECK: define void @_ZN1C4bar1Ev(%class.C* %this) nounwind align 2
void C::bar1() { }

// CHECK: define void @_ZN1C4bar2Ev(%class.C* %this) nounwind align 2
void C::bar2() { }

// CHECK: define void @_ZN1C4bar3Ev(%class.C* %this) nounwind align 1024
void C::bar3() { }
