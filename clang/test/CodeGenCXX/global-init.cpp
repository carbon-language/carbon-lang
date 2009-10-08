// RUN: clang-cc -triple=x86_64-apple-darwin9 -emit-llvm %s -o - |FileCheck %s

struct A {
  A();
  ~A();
};

// CHECK: call void @_ZN1AC1Ev
// CHECK: call i32 @__cxa_atexit(void (i8*)* bitcast (void (%struct.A*)* @_ZN1AD1Ev to void (i8*)*), i8* getelementptr inbounds (%struct.A* @a, i32 0, i32 0), i8* bitcast (i8** @__dso_handle to i8*))
A a;
