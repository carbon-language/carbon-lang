// RUN: %clang_cc1 -std=c++11 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s
// Test (r)value and CVR qualifiers on C++11 non-static member functions.
class A {
public:
  // CHECK: !"0x2e\00l\00{{.*}}\00[[@LINE+2]]"{{, [^,]+, [^,]+}}, ![[PLSR:[0-9]+]], {{.*}}[ DW_TAG_subprogram ] [line [[@LINE+2]]] [public] [reference] [l]
  // CHECK: ![[PLSR]] ={{.*}}[ DW_TAG_subroutine_type ]{{.*}}[reference]
  void l() const &;
  // CHECK: ![[ARGS:[0-9]+]] = !{null, ![[THIS:[0-9]+]]}
  // CHECK: ![[THIS]] = {{.*}} ![[CONST_A:.*]]} ; [ DW_TAG_pointer_type ]
  // CHECK: ![[CONST_A]] = {{.*}} [ DW_TAG_const_type ]
  // CHECK: !"0x2e\00r\00{{.*}}\00[[@LINE+2]]"{{, [^,]+, [^,]+}}, ![[PRSR:[0-9]+]], {{.*}}[ DW_TAG_subprogram ] [line [[@LINE+2]]] [public] [rvalue reference] [r]
  // CHECK: ![[PRSR]] ={{.*}}![[ARGS]], null, null, null}{{.*}}[ DW_TAG_subroutine_type ]{{.*}}[rvalue reference]
  void r() const &&;
};

void g() {
  A a;
  // The type of pl is "void (A::*)() const &".
  // CHECK: ![[PL:[0-9]+]]} ; [ DW_TAG_auto_variable ] [pl] [line [[@LINE+2]]]
  // CHECK: ![[PLSR]], !"{{.*}}"} ; [ DW_TAG_ptr_to_member_type ]
  auto pl = &A::l;

  // CHECK: ![[PR:[0-9]+]]} ; [ DW_TAG_auto_variable ] [pr] [line [[@LINE+2]]]
  // CHECK: ![[PRSR]], !"{{.*}}"} ; [ DW_TAG_ptr_to_member_type ]
  auto pr = &A::r;
}
