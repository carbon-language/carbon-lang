// Tests for instrumentation of C++ constructors and destructors.
//
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.11.0 -x c++ %s -o - -emit-llvm -fprofile-instr-generate | FileCheck %s

struct Foo {
  Foo() {}
  Foo(int) {}
  ~Foo() {}
};

struct Bar : public Foo {
  Bar() {}
  Bar(int x) : Foo(x) {}
  ~Bar();
};

Foo foo;
Foo foo2(1);
Bar bar;

// Profile data for complete constructors and destructors must absent.

// CHECK-NOT: @__prf_nm__ZN3FooC1Ev
// CHECK-NOT: @__prf_nm__ZN3FooC1Ei
// CHECK-NOT: @__prf_nm__ZN3FooD1Ev
// CHECK-NOT: @__prf_nm__ZN3BarC1Ev
// CHECK-NOT: @__prf_nm__ZN3BarD1Ev
// CHECK-NOT: @__prf_cn__ZN3FooD1Ev
// CHECK-NOT: @__prf_dt__ZN3FooD1Ev

int main() {
}
