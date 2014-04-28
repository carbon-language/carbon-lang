// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin -emit-llvm -o - %s -std=c++1y | FileCheck %s
// expected-no-diagnostics

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

  A b = { 7, ++b.temporary };
  // CHECK: @_ZGRN21ModifyStaticTemporary1bE = private global i32 8
  // CHECK: @_ZN21ModifyStaticTemporary1bE = global {{.*}} i32* @_ZGRN21ModifyStaticTemporary1bE, i32 8

  // Can't emit all of 'c' as a constant here, so emit the initial value of
  // 'c.temporary', not the value as modified by the partial evaluation within
  // the initialization of 'c.x'.
  A c = { 10, (++c.temporary, b.x) };
  // CHECK: @_ZGRN21ModifyStaticTemporary1cE = private global i32 10
  // CHECK: @_ZN21ModifyStaticTemporary1cE = global {{.*}} zeroinitializer
}

// CHECK: @_ZGRN28VariableTemplateWithConstRef1iIvEE = linkonce_odr constant i32 5, align 4
// CHECK: @_ZN28VariableTemplateWithConstRef3useE = constant i32* @_ZGRN28VariableTemplateWithConstRef1iIvEE
namespace VariableTemplateWithConstRef {
  template <typename T>
  const int &i = 5;
  const int &use = i<void>;
}

// CHECK: @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE = linkonce_odr constant i32 1
// CHECK: @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE2 = linkonce_odr global {{.*}} { i32* @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE }
// CHECK: @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE3 = linkonce_odr constant i32 2
// CHECK: @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE4 = linkonce_odr global {{.*}} { i32* @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE3 }
// CHECK: @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE5 = linkonce_odr constant i32 3
// CHECK: @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE6 = linkonce_odr global {{.*}} { i32* @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE5 }
// CHECK: @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE7 = linkonce_odr constant i32 4
// CHECK: @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE8 = linkonce_odr global {{.*}} { i32* @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE7 }
// CHECK: @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE9 = linkonce_odr global {{.*}} { {{.*}} @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE2, {{.*}} @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE4, {{.*}} @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE6, {{.*}} @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE8 }
// CHECK: @_ZN24VariableTemplateWithPack1pE = global {{.*}} @_ZGRN24VariableTemplateWithPack1sIJLi1ELi2ELi3ELi4EEEE9
namespace VariableTemplateWithPack {
  struct A {
    const int &r;
  };
  struct S {
    A &&a, &&b, &&c, &&d;
  };
  template <int... N>
  S &&s = {A{N}...};
  S *p = &s<1, 2, 3, 4>;
}

// CHECK: __cxa_atexit({{.*}} @_ZN1BD1Ev {{.*}} @b

// CHECK: define
// CHECK-NOT: @_ZGRN21ModifyStaticTemporary1cE
// CHECK: store {{.*}} @_ZGRN21ModifyStaticTemporary1cE, {{.*}} @_ZN21ModifyStaticTemporary1cE
// CHECK: add
// CHECK: store
// CHECK: load {{.*}} @_ZN21ModifyStaticTemporary1bE
// CHECK: store {{.*}} @_ZN21ModifyStaticTemporary1cE
