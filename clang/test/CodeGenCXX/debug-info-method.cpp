// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -std=c++11 -g %s -o - | FileCheck %s
// CHECK: !"_ZTS1A"} ; [ DW_TAG_class_type ] [A]
// CHECK: !"{{.*}}\00_ZN1A3fooEiS_3$_0\00{{.*}}", {{.*}} [protected]
// CHECK: ![[THISTYPE:[0-9]+]] = {{.*}} ; [ DW_TAG_pointer_type ] {{.*}} [artificial] [from _ZTS1A]
// CHECK: [ DW_TAG_ptr_to_member_type ] [line {{[0-9]+}}, size {{[1-9][0-9]+}}, align
// CHECK: {{.*}}![[MEMFUNTYPE:[0-9]+]], !{{.*}}} ; [ DW_TAG_ptr_to_member_type ] {{.*}} [from ]
// CHECK: ![[MEMFUNTYPE]] = {{.*}}![[MEMFUNARGS:[0-9]+]], null, null, null} ; [ DW_TAG_subroutine_type ] {{.*}} [from ]
// CHECK: ![[MEMFUNARGS]] = {{.*}}, ![[THISTYPE]],
// CHECK: !"0x101\00\00{{.*}}"{{.*}} DW_TAG_arg_variable
// CHECK: !"0x101\00\00{{.*}}"{{.*}} DW_TAG_arg_variable
// CHECK: !"0x101\00\00{{.*}}"{{.*}} DW_TAG_arg_variable
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
