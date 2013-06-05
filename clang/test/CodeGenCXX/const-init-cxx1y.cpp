// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin -emit-llvm -o - %s -std=c++1y | FileCheck %s

struct A {
  constexpr A() : n(1) {}
  ~A();
  int n;
};
struct B : A {
  A a[3];
  constexpr B() {
    ++a[0].n;
    a[1].n += 2;
    a[2].n = n + a[1].n;
  }
};
B b;

// CHECK: @b = global {{.*}} i32 1, {{.*}} { i32 2 }, {{.*}} { i32 3 }, {{.*}} { i32 4 }
// CHECK-NOT: _ZN1BC

namespace ModifyStaticTemporary {
  struct A { int &&temporary; int x; };
  constexpr int f(int &r) { r *= 9; return r - 12; }
  A a = { 6, f(a.temporary) };
  // CHECK: @_ZGRN21ModifyStaticTemporary1aE = private global i32 54
  // CHECK: @_ZN21ModifyStaticTemporary1aE = global {{.*}} i32* @_ZGRN21ModifyStaticTemporary1aE, i32 42
}

// CHECK: __cxa_atexit({{.*}} @_ZN1BD1Ev {{.*}} @b
