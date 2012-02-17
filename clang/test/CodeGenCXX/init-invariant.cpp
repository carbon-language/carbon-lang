// RUN: %clang_cc1 -triple i686-linux-gnu -emit-llvm %s -o - | FileCheck %s

// Check that we add an llvm.invariant.start to mark when a global becomes
// read-only. If globalopt can fold the initializer, it will then mark the
// variable as constant.

struct A {
  A() : n(42) {}
  int n;
};

// CHECK: @a = global {{.*}} zeroinitializer
extern const A a = A();

struct B {
  B() : n(76) {}
  mutable int n;
};

// CHECK: @b = global {{.*}} zeroinitializer
extern const B b = B();

struct C {
  C() : n(81) {}
  ~C();
  int n;
};

// CHECK: @c = global {{.*}} zeroinitializer
extern const C c = C();

int f() { return 5; }
// CHECK: @d = global i32 0
extern const int d = f();

void e() {
  static const A a = A();
}

// CHECK: define internal void @__cxx_global_var_init
// CHECK: call void @_ZN1AC1Ev({{.*}}* @a)
// CHECK-NEXT: call {{.*}}@llvm.invariant.start(i64 -1, i8* bitcast ({{.*}} @a to i8*))

// CHECK: define internal void @__cxx_global_var_init
// CHECK: call void @_ZN1BC1Ev({{.*}}* @b)
// CHECK-NOT: call {{.*}}@llvm.invariant.start(i64 -1, i8* bitcast ({{.*}} @b to i8*))

// CHECK: define internal void @__cxx_global_var_init
// CHECK: call void @_ZN1CC1Ev({{.*}}* @c)
// CHECK-NOT: call {{.*}}@llvm.invariant.start(i64 -1, i8* bitcast ({{.*}} @c to i8*))

// CHECK: define internal void @__cxx_global_var_init
// CHECK: call i32 @_Z1fv(
// CHECK: store {{.*}}, i32* @d
// CHECK: call {{.*}}@llvm.invariant.start(i64 -1, i8* bitcast ({{.*}} @d to i8*))

// CHECK: define void @_Z1ev(
// CHECK: call void @_ZN1AC1Ev(%struct.A* @_ZZ1evE1a)
// CHECK: call {{.*}}@llvm.invariant.start(i64 -1, i8* bitcast ({{.*}} @_ZZ1evE1a to i8*))
// CHECK-NOT: llvm.invariant.end
