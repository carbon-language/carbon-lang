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

// CHECK: @lsh_overflow
int lsh_overflow(int a, int b) {
  // CHECK: %[[INBOUNDS:.*]] = icmp ule i32 %[[RHS:.*]], 31
  // CHECK-NEXT: br i1 %[[INBOUNDS]]

  // CHECK: %[[SHIFTED_OUT_WIDTH:.*]] = sub nuw nsw i32 31, %[[RHS]]
  // CHECK-NEXT: %[[SHIFTED_OUT:.*]] = lshr i32 %[[LHS:.*]], %[[SHIFTED_OUT_WIDTH]]
  // CHECK-NEXT: %[[NO_OVERFLOW:.*]] = icmp eq i32 %[[SHIFTED_OUT]], 0
  // CHECK-NEXT: br i1 %[[NO_OVERFLOW]]

  // CHECK: %[[RET:.*]] = shl i32 %[[LHS]], %[[RHS]]
  // CHECK-NEXT: ret i32 %[[RET]]
  return a << b;
}

// CHECK: @rsh_inbounds
int rsh_inbounds(int a, int b) {
  // CHECK: %[[INBOUNDS:.*]] = icmp ult i32 %[[RHS:.*]], 32
  // CHECK: br i1 %[[INBOUNDS]]

  // CHECK: %[[RET:.*]] = ashr i32 %[[LHS]], %[[RHS]]
  // CHECK-NEXT: ret i32 %[[RET]]
  return a >> b;
}

// CHECK: @no_return
int no_return() {
  // Reaching the end of a noreturn function is fine in C.
  // CHECK-NOT: call
  // CHECK-NOT: unreachable
  // CHECK: ret i32
}
