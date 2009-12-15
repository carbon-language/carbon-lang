// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin9 | FileCheck %s

struct I {
  int i;
  I();
  ~I();
};

void g(int);

volatile int i;

void f1() {
  // CHECK: call void @_ZN1IC1Ev
  g(i ? I().i : 0);
  // CHECK: call void @_Z1gi
  // CHECK: call void @_ZN1ID1Ev

  // CHECK: call void @_ZN1IC1Ev
  g(i || I().i);
  // CHECK: call void @_Z1gi
  // CHECK: call void @_ZN1ID1Ev

  // CHECK: call void @_ZN1IC1Ev
  g(i && I().i);
  // CHECK: call void @_Z1gi
  // CHECK: call void @_ZN1ID1Ev
}
