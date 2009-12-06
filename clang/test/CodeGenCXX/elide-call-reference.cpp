// RUN: clang-cc %s -emit-llvm -o - | FileCheck %s
// PR5695

struct A { A(const A&); ~A(); };
A& a();
void b() {
  A x = a();
}

// CHECK: call void @_ZN1AC1ERKS_
// CHECK: call void @_ZN1AD1Ev
