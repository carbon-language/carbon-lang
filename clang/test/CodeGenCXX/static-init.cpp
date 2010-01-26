// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// CHECK: @_ZZ1hvE1i = internal global i32 0, align 4
struct A {
  A();
  ~A();
};

void f() {
  // CHECK: call void @_ZN1AC1Ev(
  // CHECK: call i32 @__cxa_atexit(void (i8*)* bitcast (void (%struct.A*)* @_ZN1AD1Ev to void (i8*)*), i8* getelementptr inbounds (%struct.A* @_ZZ1fvE1a, i32 0, i32 0), i8* bitcast (i8** @__dso_handle to i8*))
  static A a;
}

void g() {
  // CHECK: call noalias i8* @_Znwm(i64 1)
  // CHECK: call void @_ZN1AC1Ev(
  static A& a = *new A;
}

int a();
void h() {
  static const int i = a();
}
