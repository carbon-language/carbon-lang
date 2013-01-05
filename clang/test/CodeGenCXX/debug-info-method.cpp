// RUN: %clang_cc1 -emit-llvm -g %s -o - | FileCheck %s
// CHECK: metadata !"_ZN1A3fooEi", {{.*}}, i32 258
// CHECK: ""{{.*}}DW_TAG_arg_variable
class A {
protected:
  void foo(int);
}; 

void A::foo(int) {
}

A a;
