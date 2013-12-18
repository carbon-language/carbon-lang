// RUN: %clang_cc1 -std=c++11 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s
// Test (r)value qualifiers on C++11 non-static member functions.
class A {
public:
  // CHECK: [ DW_TAG_subprogram ] [line [[@LINE+1]]] [reference] [l]
  void l() const &;
  // CHECK: [ DW_TAG_subprogram ] [line [[@LINE+1]]] [rvalue reference] [r]
  void r() const &&;
};

void g() {
  A a;
  // The type of pl is "void (A::*)() const &".
  // CHECK: metadata ![[PL:[0-9]+]], i32 0, i32 0} ; [ DW_TAG_auto_variable ] [pl] [line [[@LINE+3]]]
  // CHECK: metadata ![[PLSR:[0-9]+]], metadata !"{{.*}}"} ; [ DW_TAG_ptr_to_member_type ]
  // CHECK: ![[PLSR]] ={{.*}}[ DW_TAG_subroutine_type ]{{.*}}[reference]
  auto pl = &A::l;

  // CHECK: metadata ![[PR:[0-9]+]], i32 0, i32 0} ; [ DW_TAG_auto_variable ] [pr] [line [[@LINE+3]]]
  // CHECK: metadata ![[PRSR:[0-9]+]], metadata !"{{.*}}"} ; [ DW_TAG_ptr_to_member_type ]
  // CHECK: ![[PRSR]] ={{.*}}[ DW_TAG_subroutine_type ]{{.*}}[rvalue reference]
  auto pr = &A::r;
}
