// RUN: %clang_cc1 -triple=x86_64-apple-darwin10 -emit-llvm -fexceptions %s -o - |FileCheck %s
// RUN: %clang_cc1 -triple=x86_64-apple-darwin10 -emit-llvm %s -o - |FileCheck -check-prefix NOEXC %s

struct A {
  A();
  ~A();
};

struct B { B(); ~B(); };

struct C { void *field; };

struct D { ~D(); };

// CHECK: @__dso_handle = external unnamed_addr global i8
// CHECK: @c = global %struct.C zeroinitializer, align 8

// It's okay if we ever implement the IR-generation optimization to remove this.
// CHECK: @_ZN5test3L3varE = internal constant i8* getelementptr inbounds ([7 x i8]* 

// PR6205: The casts should not require global initializers
// CHECK: @_ZN6PR59741cE = external global %"struct.PR5974::C"
// CHECK: @_ZN6PR59741aE = global %"struct.PR5974::A"* getelementptr inbounds (%"struct.PR5974::C"* @_ZN6PR59741cE, i32 0, i32 0)
// CHECK: @_ZN6PR59741bE = global %"struct.PR5974::B"* bitcast (i8* getelementptr (i8* bitcast (%"struct.PR5974::C"* @_ZN6PR59741cE to i8*), i64 4) to %"struct.PR5974::B"*), align 8

// CHECK: call void @_ZN1AC1Ev(%struct.A* @a)
// CHECK: call i32 @__cxa_atexit(void (i8*)* bitcast (void (%struct.A*)* @_ZN1AD1Ev to void (i8*)*), i8* getelementptr inbounds (%struct.A* @a, i32 0, i32 0), i8* @__dso_handle)
A a;

// CHECK: call void @_ZN1BC1Ev(%struct.B* @b)
// CHECK: call i32 @__cxa_atexit(void (i8*)* bitcast (void (%struct.B*)* @_ZN1BD1Ev to void (i8*)*), i8* getelementptr inbounds (%struct.B* @b, i32 0, i32 0), i8* @__dso_handle)
B b;

// PR6205: this should not require a global initializer
// CHECK-NOT: call void @_ZN1CC1Ev(%struct.C* @c)
C c;

// CHECK: call i32 @__cxa_atexit(void (i8*)* bitcast (void (%struct.D*)* @_ZN1DD1Ev to void (i8*)*), i8* getelementptr inbounds (%struct.D* @d, i32 0, i32 0), i8* @__dso_handle)
D d;

// <rdar://problem/7458115>
namespace test1 {
  int f();
  const int x = f();   // This has side-effects and gets emitted immediately.
  const int y = x - 1; // This gets deferred.
  const int z = ~y;    // This also gets deferred, but gets "undeferred" before y.
  int test() { return z; }
// CHECK:      define i32 @_ZN5test14testEv()

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

namespace test3 {
  // Tested at the beginning of the file.
  const char * const var = "string";
  extern const char * const var;

  const char *test() { return var; }
}

namespace test6 {
  struct A {
    A();
  };
  extern int foo();

  // This needs an initialization function and guard variables.
  // CHECK: load i8* bitcast (i64* @_ZGVN5test61xE
  // CHECK: [[CALL:%.*]] = call i32 @_ZN5test63fooEv
  // CHECK-NEXT: store i32 [[CALL]], i32* @_ZN5test61xE
  // CHECK-NEXT: store i64 1, i64* @_ZGVN5test61xE
  __attribute__((weak)) int x = foo();
}

namespace PR5974 {
  struct A { int a; };
  struct B { int b; };
  struct C : A, B { int c; };

  extern C c;

  // These should not require global initializers.
  A* a = &c;
  B* b = &c;
}
// CHECK:      define internal void [[TEST1_Z_INIT:@.*]]()
// CHECK:        load i32* @_ZN5test1L1yE
// CHECK-NEXT:   xor
// CHECK-NEXT:   store i32 {{.*}}, i32* @_ZN5test1L1zE
// CHECK:      define internal void [[TEST1_Y_INIT:@.*]]()
// CHECK:        load i32* @_ZN5test1L1xE
// CHECK-NEXT:   sub
// CHECK-NEXT:   store i32 {{.*}}, i32* @_ZN5test1L1yE

// PR9570: the indirect field shouldn't crash IR gen.
namespace test5 {
  static union {
    unsigned bar[4096] __attribute__((aligned(128)));
  };
}


// At the end of the file, we check that y is initialized before z.

// CHECK: define internal void @_GLOBAL__I_a() section "__TEXT,__StaticInit,regular,pure_instructions" {
// CHECK:   call void [[TEST1_Y_INIT]]
// CHECK:   call void [[TEST1_Z_INIT]]

// rdar://problem/8090834: this should be nounwind
// CHECK-NOEXC: define internal void @_GLOBAL__I_a() nounwind section "__TEXT,__StaticInit,regular,pure_instructions" {
