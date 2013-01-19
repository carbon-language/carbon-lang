// RUN: %clang_cc1 -emit-llvm -std=c++11 -g %s -o - | FileCheck %s
// CHECK: ![[THISTYPE:[0-9]+]] = {{.*}} ; [ DW_TAG_pointer_type ] {{.*}} [artificial] [from A]
// CHECK: metadata !"_ZN1A3fooEiS_3$_0", {{.*}} [protected]
// CHECK: DW_TAG_ptr_to_member_type
// CHECK: {{.*}}metadata ![[MEMFUNTYPE:[0-9]+]], metadata !{{.*}}} ; [ DW_TAG_ptr_to_member_type ] {{.*}} [from ]
// CHECK: ![[MEMFUNTYPE]] = {{.*}}metadata ![[MEMFUNARGS:[0-9]+]], i32 0, i32 0} ; [ DW_TAG_subroutine_type ] {{.*}} [from ]
// CHECK: ![[MEMFUNARGS]] = {{.*}}, metadata ![[THISTYPE]],
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

int A::*x = 0;
int (A::*y)(int) = 0;
