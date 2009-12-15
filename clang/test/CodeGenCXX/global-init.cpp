// RUN: %clang_cc1 -triple=x86_64-apple-darwin10 -emit-llvm %s -o - |FileCheck %s

struct A {
  A();
  ~A();
};

struct B { B(); ~B(); };

// CHECK: call void @_ZN1AC1Ev(%struct.A* @a)
// CHECK: call i32 @__cxa_atexit(void (i8*)* bitcast (void (%struct.A*)* @_ZN1AD1Ev to void (i8*)*), i8* getelementptr inbounds (%struct.A* @a, i32 0, i32 0), i8* bitcast (i8** @__dso_handle to i8*))
A a;

// CHECK: call void @_ZN1BC1Ev(%struct.A* @b)
// CHECK: call i32 @__cxa_atexit(void (i8*)* bitcast (void (%struct.A*)* @_ZN1BD1Ev to void (i8*)*), i8* getelementptr inbounds (%struct.A* @b, i32 0, i32 0), i8* bitcast (i8** @__dso_handle to i8*))
B b;
