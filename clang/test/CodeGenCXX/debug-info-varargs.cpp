// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -g %s -o - | FileCheck %s

struct A
{
  // CHECK-DAG: ", i32 [[@LINE+1]], metadata ![[ATY:[0-9]+]]{{.*}}[ DW_TAG_subprogram ]{{.*}}[a]
  void a(int c, ...) {}
  // CHECK: ![[ATY]] ={{.*}} metadata ![[AARGS:[0-9]+]], i32 0, null, null, null} ; [ DW_TAG_subroutine_type ]
  // CHECK: ![[AARGS]] = {{.*}} metadata ![[UNSPEC:[0-9]+]]}
  // CHECK: ![[UNSPEC]] = {{.*}} [ DW_TAG_unspecified_parameters ]
};

  // CHECK: ", i32 [[@LINE+1]], metadata ![[BTY:[0-9]+]]{{.*}}[ DW_TAG_subprogram ]{{.*}}[b]
void b(int c, ...) {
  // CHECK: ![[BTY]] ={{.*}} metadata ![[BARGS:[0-9]+]], i32 0, null, null, null} ; [ DW_TAG_subroutine_type ]
  // CHECK: ![[BARGS]] = {{.*}} metadata ![[UNSPEC:[0-9]+]]}

  A a;

  // CHECK: metadata ![[PST:[0-9]+]], i32 0, i32 0} ; [ DW_TAG_auto_variable ] [fptr] [line [[@LINE+1]]]
  void (*fptr)(int, ...) = b;
  // CHECK: ![[PST]] ={{.*}} metadata ![[BTY]]} ; [ DW_TAG_pointer_type ]
}
