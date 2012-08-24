// RUN: %clang_cc1 -fcatch-undefined-behavior -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s

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
  // CHECK: %[[SIZE:.*]] = call i64 @llvm.objectsize.i64
  // CHECK-NEXT: icmp uge i64 %[[SIZE]], 4

  // CHECK: %[[PTRINT:.*]] = ptrtoint
  // CHECK-NEXT: %[[MISALIGN:.*]] = and i64 %[[PTRINT]], 3
  // CHECK-NEXT: icmp eq i64 %[[MISALIGN]], 0
  return *a;
}
