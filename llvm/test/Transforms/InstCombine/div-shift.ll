; RUN: opt < %s -instcombine -S | FileCheck %s
; XFAIL: *

define i32 @t1(i16 zeroext %x, i32 %y) nounwind {
entry:
; CHECK: t1
; CHECK-NOT: sdiv
; CHECK: lshr i32 %conv
  %conv = zext i16 %x to i32
  %s = shl i32 2, %y
  %d = sdiv i32 %conv, %s
  ret i32 %d
}

; rdar://11721329
define i64 @t2(i64 %x, i32 %y) nounwind  {
; CHECK: t2
; CHECK-NOT: udiv
; CHECK: lshr i64 %x
  %1 = shl i32 1, %y
  %2 = zext i32 %1 to i64
  %3 = udiv i64 %x, %2
  ret i64 %3
}

; PR13250
define i64 @t3(i64 %x, i32 %y) nounwind  {
; CHECK: t3
; CHECK-NOT: udiv
; CHECK-NEXT: %1 = add i32 %y, 2
; CHECK-NEXT: %2 = zext i32 %1 to i64
; CHECK-NEXT: %3 = lshr i64 %x, %2
; CHECK-NEXT: ret i64 %3
  %1 = shl i32 4, %y
  %2 = zext i32 %1 to i64
  %3 = udiv i64 %x, %2
  ret i64 %3
}

define i32 @t4(i32 %x, i32 %y) nounwind {
; CHECK: t4
; CHECK-NOT: udiv
; CHECK-NEXT: [[CMP:%.*]] = icmp ult i32 %y, 5
; CHECK-NEXT: [[SEL:%.*]] = select i1 [[CMP]], i32 5, i32 %y
; CHECK-NEXT: [[SHR:%.*]] = lshr i32 %x, [[SEL]]
; CHECK-NEXT: ret i32 [[SHR]]
  %1 = shl i32 1, %y
  %2 = icmp ult i32 %1, 32
  %3 = select i1 %2, i32 32, i32 %1
  %4 = udiv i32 %x, %3
  ret i32 %4
}

define i32 @t5(i1 %x, i1 %y, i32 %V) nounwind {
; CHECK: t5
; CHECK-NOT: udiv
; CHECK-NEXT: [[SEL1:%.*]] = select i1 %x, i32 5, i32 6
; CHECK-NEXT: [[SEL2:%.*]] = select i1 %y, i32 [[SEL1]], i32 %V
; CHECK-NEXT: [[LSHR:%.*]] = lshr i32 %V, [[SEL2]]
; CHECK-NEXT: ret i32 [[LSHR]]
  %1 = shl i32 1, %V
  %2 = select i1 %x, i32 32, i32 64
  %3 = select i1 %y, i32 %2, i32 %1
  %4 = udiv i32 %V, %3
  ret i32 %4
}

define i32 @t6(i32 %x, i32 %z) nounwind{
; CHECK: t6
; CHECK-NEXT: [[CMP:%.*]] = icmp eq i32 %x, 0
; CHECK-NOT: udiv i32 %z, %x
  %x_is_zero = icmp eq i32 %x, 0
  %divisor = select i1 %x_is_zero, i32 1, i32 %x
  %y = udiv i32 %z, %divisor
  ret i32 %y
}
