// RUN: %clang_cc1 -emit-llvm -std=c++11 -g %s -o - | FileCheck %s
// CHECK: metadata !"_ZN1A3fooEiS_3$_0", {{.*}}, i32 258
// CHECK: ""{{.*}}DW_TAG_arg_variable
// CHECK: ""{{.*}}DW_TAG_arg_variable
// CHECK: ""{{.*}}DW_TAG_arg_variable
union {
  int a;
  float b;
} u;

class A {
protected:
  void foo(int, A, decltype(u));
}; 

void A::foo(int, A, decltype(u)) {
}

A a;
