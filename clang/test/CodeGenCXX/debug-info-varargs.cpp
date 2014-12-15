// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -g %s -o - | FileCheck %s

struct A
{
  // CHECK-DAG:  !"0x2e\00a\00a\00_ZN1A1aEiz\00[[@LINE+1]]\00{{[^,]*}}"{{, [^,]+, [^,]+}}, ![[ATY:[0-9]+]]{{.*}}[ DW_TAG_subprogram ]{{.*}}[a]
  void a(int c, ...) {}
  // CHECK: ![[ATY]] ={{.*}} ![[AARGS:[0-9]+]], null, null, null} ; [ DW_TAG_subroutine_type ]
  // We no longer use an explicit unspecified parameter. Instead we use a trailing null to mean the function is variadic.
  // CHECK: ![[AARGS]] = !{null, !{{[0-9]+}}, !{{[0-9]+}}, null}
};

  // CHECK:  !"0x2e\00b\00b\00_Z1biz\00[[@LINE+1]]\00{{[^,]*}}"{{, [^,]+, [^,]+}}, ![[BTY:[0-9]+]]{{.*}}[ DW_TAG_subprogram ]{{.*}}[b]
void b(int c, ...) {
  // CHECK: ![[BTY]] ={{.*}} ![[BARGS:[0-9]+]], null, null, null} ; [ DW_TAG_subroutine_type ]
  // CHECK: ![[BARGS]] = !{null, !{{[0-9]+}}, null}

  A a;

  // CHECK:  !"0x100\00fptr\00[[@LINE+1]]\000"{{, [^,]+, [^,]+}}, ![[PST:[0-9]+]]} ; [ DW_TAG_auto_variable ] [fptr] [line [[@LINE+1]]]
  void (*fptr)(int, ...) = b;
  // CHECK: ![[PST]] ={{.*}} ![[BTY]]} ; [ DW_TAG_pointer_type ]
}
