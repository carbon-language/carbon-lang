// RUN: %clang_cc1 -fcatch-undefined-behavior -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s

// CHECK: @_Z17reference_binding
void reference_binding(int *p) {
  // C++ core issue 453: If an lvalue to which a reference is directly bound
  // designates neither an existing object or function of an appropriate type,
  // nor a region of storage of suitable size and alignment to contain an object
  // of the reference's type, the behavior is undefined.

  // CHECK: icmp ne {{.*}}, null

  // CHECK: %[[SIZE:.*]] = call i64 @llvm.objectsize.i64
  // CHECK-NEXT: icmp uge i64 %[[SIZE]], 4

  // CHECK: %[[PTRINT:.*]] = ptrtoint
  // CHECK-NEXT: %[[MISALIGN:.*]] = and i64 %[[PTRINT]], 3
  // CHECK-NEXT: icmp eq i64 %[[MISALIGN]], 0
  int &r = *p;
}

struct S {
  double d;
  int a, b;
  virtual int f();
};

// CHECK: @_Z13member_access
void member_access(S *p) {
  // (1) Check 'p' is appropriately sized and aligned for member access.

  // FIXME: Check vptr is for 'S' or a class derived from 'S'.

  // CHECK: icmp ne {{.*}}, null

  // CHECK: %[[SIZE:.*]] = call i64 @llvm.objectsize.i64
  // CHECK-NEXT: icmp uge i64 %[[SIZE]], 24

  // CHECK: %[[PTRINT:.*]] = ptrtoint
  // CHECK-NEXT: %[[MISALIGN:.*]] = and i64 %[[PTRINT]], 7
  // CHECK-NEXT: icmp eq i64 %[[MISALIGN]], 0

  // (2) Check 'p->b' is appropriately sized and aligned for a load.

  // FIXME: Suppress this in the trivial case of a member access, because we
  // know we've just checked the member access expression itself.

  // CHECK: %[[SIZE:.*]] = call i64 @llvm.objectsize.i64
  // CHECK-NEXT: icmp uge i64 %[[SIZE]], 4

  // CHECK: %[[PTRINT:.*]] = ptrtoint
  // CHECK-NEXT: %[[MISALIGN:.*]] = and i64 %[[PTRINT]], 3
  // CHECK-NEXT: icmp eq i64 %[[MISALIGN]], 0
  int k = p->b;

  // (3) Check 'p' is appropriately sized and aligned for member function call.

  // FIXME: Check vptr is for 'S' or a class derived from 'S'.

  // CHECK: icmp ne {{.*}}, null

  // CHECK: %[[SIZE:.*]] = call i64 @llvm.objectsize.i64
  // CHECK-NEXT: icmp uge i64 %[[SIZE]], 24

  // CHECK: %[[PTRINT:.*]] = ptrtoint
  // CHECK-NEXT: %[[MISALIGN:.*]] = and i64 %[[PTRINT]], 7
  // CHECK-NEXT: icmp eq i64 %[[MISALIGN]], 0
  k = p->f();
}

// CHECK: @_Z12lsh_overflow
int lsh_overflow(int a, int b) {
  // CHECK: %[[INBOUNDS:.*]] = icmp ule i32 %[[RHS:.*]], 31
  // CHECK-NEXT: br i1 %[[INBOUNDS]]

  // CHECK: %[[SHIFTED_OUT_WIDTH:.*]] = sub nuw nsw i32 31, %[[RHS]]
  // CHECK-NEXT: %[[SHIFTED_OUT:.*]] = lshr i32 %[[LHS:.*]], %[[SHIFTED_OUT_WIDTH]]

  // This is present for C++11 but not for C: C++ core issue 1457 allows a '1'
  // to be shifted into the sign bit, but not out of it.
  // CHECK-NEXT: %[[SHIFTED_OUT_NOT_SIGN:.*]] = lshr i32 %[[SHIFTED_OUT]], 1

  // CHECK-NEXT: %[[NO_OVERFLOW:.*]] = icmp eq i32 %[[SHIFTED_OUT_NOT_SIGN]], 0
  // CHECK-NEXT: br i1 %[[NO_OVERFLOW]]

  // CHECK: %[[RET:.*]] = shl i32 %[[LHS]], %[[RHS]]
  // CHECK-NEXT: ret i32 %[[RET]]
  return a << b;
}

// CHECK: @_Z9no_return
int no_return() {
  // CHECK: call void @llvm.trap
  // CHECK: unreachable
}
