// RUN: %clang_cc1 -fcatch-undefined-behavior -emit-llvm %s -o - | FileCheck %s

// PR6805
// CHECK: @foo
void foo() {
  union { int i; } u;
  // CHECK: objectsize
  // CHECK: icmp uge
  u.i=1;
}

// CHECK: @bar
int bar(int *a) {
  // CHECK: objectsize
  // CHECK: icmp uge
  return *a;
}
