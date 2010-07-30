// RUN: %clang_cc1 -triple=x86_64-apple-darwin10 -emit-llvm -fexceptions %s -o - |FileCheck %s
// RUN: %clang_cc1 -triple=x86_64-apple-darwin10 -emit-llvm %s -o - |FileCheck -check-prefix NOEXC %s

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

// <rdar://problem/7458115>
namespace test1 {
  int f();
  const int x = f();   // This has side-effects and gets emitted immediately.
  const int y = x - 1; // This gets deferred.
  const int z = ~y;    // This also gets deferred, but gets "undeferred" before y.
  int test() { return z; }
// CHECK:      define i32 @_ZN5test14testEv() {

  // All of these initializers end up delayed, so we check them later.
}

// <rdar://problem/8246444>
namespace test2 {
  struct allocator { allocator(); ~allocator(); };
  struct A { A(const allocator &a = allocator()); ~A(); };

  A a;
// CHECK: call void @_ZN5test29allocatorC1Ev(
// CHECK: invoke void @_ZN5test21AC1ERKNS_9allocatorE(
// CHECK: call void @_ZN5test29allocatorD1Ev(
// CHECK: call i32 @__cxa_atexit({{.*}} @_ZN5test21AD1Ev {{.*}} @_ZN5test21aE
}

// CHECK:      define internal void [[TEST1_Z_INIT:@.*]]()
// CHECK:        load i32* @_ZN5test1L1yE
// CHECK-NEXT:   xor
// CHECK-NEXT:   store i32 {{.*}}, i32* @_ZN5test1L1zE
// CHECK:      define internal void [[TEST1_Y_INIT:@.*]]()
// CHECK:        load i32* @_ZN5test1L1xE
// CHECK-NEXT:   sub
// CHECK-NEXT:   store i32 {{.*}}, i32* @_ZN5test1L1yE

// At the end of the file, we check that y is initialized before z.

// CHECK: define internal void @_GLOBAL__I_a() section "__TEXT,__StaticInit,regular,pure_instructions" {
// CHECK:   call void [[TEST1_Y_INIT]]
// CHECK:   call void [[TEST1_Z_INIT]]

// rdar://problem/8090834: this should be nounwind
// CHECK-NOEXC: define internal void @_GLOBAL__I_a() nounwind section "__TEXT,__StaticInit,regular,pure_instructions" {
