// RUN: %clang_cc1 -std=c++0x -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
struct X {
  int f() &;
  int g() &&;
  int h() const &&;
};

// CHECK: define i32 @_ZNR1X1fEv
int X::f() & { return 0; }
// CHECK: define i32 @_ZNO1X1gEv
int X::g() && { return 0; }
// CHECK: define i32 @_ZNKO1X1hEv
int X::h() const && { return 0; }

// CHECK: define void @_Z1fM1XRFivEMS_OFivEMS_KOFivE
void f(int (X::*)() &, int (X::*)() &&, int (X::*)() const&&) { }
