// RUN: %clang_cc1 -std=c++11 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s
// Test (r)value and CVR qualifiers on C++11 non-static member functions.
class A {
public:
  // CHECK: i32 [[@LINE+2]], metadata ![[PLSR:[0-9]+]], {{.*}}[ DW_TAG_subprogram ] [line [[@LINE+2]]] [reference] [l]
  // CHECK: ![[PLSR]] ={{.*}}[ DW_TAG_subroutine_type ]{{.*}}[reference]
  void l() const &;
  // CHECK: ![[ARGS:[0-9]+]] = metadata !{null, metadata ![[THIS:[0-9]+]]}
  // CHECK: ![[THIS]] = {{.*}} metadata ![[CONST_A:.*]]} ; [ DW_TAG_pointer_type ]
  // CHECK: ![[CONST_A]] = {{.*}} [ DW_TAG_const_type ]
  // CHECK: i32 [[@LINE+2]], metadata ![[PRSR:[0-9]+]], {{.*}}[ DW_TAG_subprogram ] [line [[@LINE+2]]] [rvalue reference] [r]
  // CHECK: ![[PRSR]] ={{.*}}metadata ![[ARGS]], i32 0, null, null, null}{{.*}}[ DW_TAG_subroutine_type ]{{.*}}[rvalue reference]
  void r() const &&;
};

void g() {
  A a;
  // The type of pl is "void (A::*)() const &".
  // CHECK: metadata ![[PL:[0-9]+]], i32 0, i32 0} ; [ DW_TAG_auto_variable ] [pl] [line [[@LINE+2]]]
  // CHECK: metadata ![[PLSR]], metadata !"{{.*}}"} ; [ DW_TAG_ptr_to_member_type ]
  auto pl = &A::l;

  // CHECK: metadata ![[PR:[0-9]+]], i32 0, i32 0} ; [ DW_TAG_auto_variable ] [pr] [line [[@LINE+2]]]
  // CHECK: metadata ![[PRSR]], metadata !"{{.*}}"} ; [ DW_TAG_ptr_to_member_type ]
  auto pr = &A::r;
}
