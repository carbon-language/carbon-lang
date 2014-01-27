// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -Wno-unreachable-code -Werror -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

int val = 42;
int& test1() {
  return throw val, val;
}

int test2() {
  return val ? throw val : val;
}

// rdar://problem/8608801
void test3() {
  throw false;
}

// PR10582
int test4() {
  return 1 ? throw val : val;
}

// PR15923
int test5(bool x, bool y, int z) {
  return (x ? throw 1 : y) ? z : throw 2;
}
// CHECK-LABEL: define i32 @_Z5test5bbi(
// CHECK: br i1
//
// x.true:
// CHECK: call void @__cxa_throw(
// CHECK-NEXT: unreachable
//
// x.false:
// CHECK: br i1
//
// y.true:
// CHECK: load i32*
// CHECK: br label
//
// y.false:
// CHECK: call void @__cxa_throw(
// CHECK-NEXT: unreachable
//
// end:
// CHECK: ret i32

int test6(bool x, bool y, int z) {
  return (x ? throw 1 : y) ? z : (throw 2);
}
// CHECK-LABEL: define i32 @_Z5test6bbi(
// CHECK: br i1
//
// x.true:
// CHECK: call void @__cxa_throw(
// CHECK-NEXT: unreachable
//
// x.false:
// CHECK: br i1
//
// y.true:
// CHECK: load i32*
// CHECK: br label
//
// y.false:
// CHECK: call void @__cxa_throw(
// CHECK-NEXT: unreachable
//
// end:
// CHECK: ret i32

namespace DR1560 {
  struct A {
    ~A();
  };
  extern bool b;
  A get();
  // CHECK-LABEL: @_ZN6DR15601bE
  const A &r = b ? get() : throw 0;
  // CHECK-NOT: call {{.*}}@_ZN6DR15601AD1Ev
  // CHECK: call {{.*}} @__cxa_atexit({{.*}} @_ZN6DR15601AD1Ev {{.*}} @_ZGRN6DR15601rE
  // CHECK-NOT: call {{.*}}@_ZN6DR15601AD1Ev
}
