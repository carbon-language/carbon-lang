// RUN: %clang_cc1 -triple x86_64-apple-darwin -g -emit-llvm %s -o - | FileCheck %s

// CHECK: ![[B:.*]] = {{.*}}, null, null, ![[BMEMBERS:.*]], null, null, null} ; [ DW_TAG_structure_type ] [B] [line [[@LINE+1]], size 8, align 8, offset 0] [def] [from ]
struct B {
  B(struct A *);
// CHECK: ![[BMEMBERS]] = !{![[BB:.*]]}
// CHECK: ![[BB]] = {{.*}} ![[B]], ![[TY:[0-9]+]], {{.*}}} ; [ DW_TAG_subprogram ] [line [[@LINE-2]]] [B]
// CHECK: ![[TY]] = {{.*}} ![[ARGS:[0-9]+]], null, null, null} ; [ DW_TAG_subroutine_type ]
// CHECK: ![[ARGS]] = !{null, ![[THIS:[0-9]+]],
// CHECK: ![[THIS]] = {{.*}}[[B]]} ; [ DW_TAG_pointer_type ] [
};

struct C {
 B b;
 C(struct A *);
 virtual ~C();
};

C::C(struct A *a)
  : b(a) {
}
