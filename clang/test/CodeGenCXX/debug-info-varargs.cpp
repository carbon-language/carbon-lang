// RUN: %clang_cc1 -emit-llvm -g %s -o - | FileCheck %s

struct A
{
  // CHECK-DAG: ", i32 [[@LINE+1]], metadata ![[ATY:[0-9]+]]{{.*}}[ DW_TAG_subprogram ]{{.*}}[a]
  void a(int c, ...) {}
  // CHECK-DAG: ![[ATY]] ={{.*}} metadata ![[AARGS:[0-9]+]], i32 0, null, null, null} ; [ DW_TAG_subroutine_type ]
  // CHECK-DAG: ![[AARGS]] = {{.*}} metadata ![[UNSPEC:[0-9]+]]}
  // CHECK-DAG: ![[UNSPEC]] = metadata !{i32 786456}
};

  // CHECK-DAG: ", i32 [[@LINE+1]], metadata ![[BTY:[0-9]+]]{{.*}}[ DW_TAG_subprogram ]{{.*}}[b]
void b(int c, ...) {
  // CHECK-DAG: ![[BTY]] ={{.*}} metadata ![[BARGS:[0-9]+]], i32 0, null, null, null} ; [ DW_TAG_subroutine_type ]
  // CHECK-DAG: ![[BARGS]] = {{.*}} metadata ![[UNSPEC:[0-9]+]]}

  A a;

  // CHECK-DAG: metadata ![[PST:[0-9]+]], i32 0, i32 0} ; [ DW_TAG_auto_variable ] [fptr] [line [[@LINE+1]]]
  void (*fptr)(int, ...) = b;
  // CHECK-DAG: ![[PST]] ={{.*}} metadata ![[ST:[0-9]+]]} ; [ DW_TAG_pointer_type ]
  // CHECK-DAG: ![[ST]] ={{.*}} metadata ![[ARGS:[0-9]+]], i32 0, null, null, null} ; [ DW_TAG_subroutine_type ]
  // CHECK-DAG: ![[ARGS]] = {{.*}} metadata ![[UNSPEC]]}
}
