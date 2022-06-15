// RUN: %clang_cc1 -no-opaque-pointers -triple i686-linux-gnu -emit-llvm %s -disable-llvm-passes -o - | FileCheck %s --check-prefix=CHECK-O0
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-linux-gnu -emit-llvm %s -O1 -disable-llvm-passes -o - | FileCheck %s

// Check that we add an llvm.invariant.start.p0i8 to mark when a global becomes
// read-only. If globalopt can fold the initializer, it will then mark the
// variable as constant.

// Do not produce markers at -O0.
// CHECK-O0-NOT: llvm.invariant.start.p0i8

struct A {
  A();
  int n;
};

// CHECK: @a ={{.*}} global {{.*}} zeroinitializer
extern const A a = A();

struct B {
  B();
  mutable int n;
};

// CHECK: @b ={{.*}} global {{.*}} zeroinitializer
extern const B b = B();

struct C {
  C();
  ~C();
  int n;
};

// CHECK: @c ={{.*}} global {{.*}} zeroinitializer
extern const C c = C();

int f();
// CHECK: @d ={{.*}} global i32 0
extern const int d = f();

void e() {
  static const A a = A();
}

// CHECK: call void @_ZN1AC1Ev({{.*}}* noundef {{[^,]*}} @a)
// CHECK: call {{.*}}@llvm.invariant.start.p0i8(i64 4, i8* bitcast ({{.*}} @a to i8*))

// CHECK: call void @_ZN1BC1Ev({{.*}}* noundef {{[^,]*}} @b)
// CHECK-NOT: call {{.*}}@llvm.invariant.start.p0i8(i64 noundef 4, i8* bitcast ({{.*}} @b to i8*))

// CHECK: call void @_ZN1CC1Ev({{.*}}* noundef {{[^,]*}} @c)
// CHECK-NOT: call {{.*}}@llvm.invariant.start.p0i8(i64 noundef 4, i8* bitcast ({{.*}} @c to i8*))

// CHECK: call noundef i32 @_Z1fv(
// CHECK: store {{.*}}, i32* @d
// CHECK: call {{.*}}@llvm.invariant.start.p0i8(i64 4, i8* bitcast ({{.*}} @d to i8*))

// CHECK-LABEL: define{{.*}} void @_Z1ev(
// CHECK: call void @_ZN1AC1Ev(%struct.A* noundef {{[^,]*}} @_ZZ1evE1a)
// CHECK: call {{.*}}@llvm.invariant.start.p0i8(i64 4, i8* {{.*}}bitcast ({{.*}} @_ZZ1evE1a to i8*))
// CHECK-NOT: llvm.invariant.end
