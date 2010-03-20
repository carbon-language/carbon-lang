// RUN: %clang_cc1 -triple=x86_64-apple-darwin10 -emit-llvm %s -o - |FileCheck %s

struct A {
  A();
  ~A();
};

struct B { B(); ~B(); };

struct C { void *field; };

struct D { ~D(); };

// CHECK: @c = global %struct.C zeroinitializer, align 8

// CHECK: call void @_ZN1AC1Ev(%struct.A* @a)
// CHECK: call i32 @__cxa_atexit(void (i8*)* bitcast (void (%struct.A*)* @_ZN1AD1Ev to void (i8*)*), i8* getelementptr inbounds (%struct.A* @a, i32 0, i32 0), i8* bitcast (i8** @__dso_handle to i8*))
A a;

// CHECK: call void @_ZN1BC1Ev(%struct.A* @b)
// CHECK: call i32 @__cxa_atexit(void (i8*)* bitcast (void (%struct.A*)* @_ZN1BD1Ev to void (i8*)*), i8* getelementptr inbounds (%struct.A* @b, i32 0, i32 0), i8* bitcast (i8** @__dso_handle to i8*))
B b;

// PR6205: this should not require a global initializer
// CHECK-NOT: call void @_ZN1CC1Ev(%struct.C* @c)
C c;

// CHECK: call i32 @__cxa_atexit(void (i8*)* bitcast (void (%struct.A*)* @_ZN1DD1Ev to void (i8*)*), i8* getelementptr inbounds (%struct.A* @d, i32 0, i32 0), i8* bitcast (i8** @__dso_handle to i8*))
D d;

// CHECK: define internal void @_GLOBAL__I_a() {
