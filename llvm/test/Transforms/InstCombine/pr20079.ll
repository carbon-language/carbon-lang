; RUN: opt -S -instcombine < %s | FileCheck %s
@b = internal global [1 x i32] zeroinitializer, align 4
@c = internal global i32 0, align 4

; CHECK-LABEL: @fn1
; CHECK: [[ADD:%.*]] = add i32 %a, -1
; CHECK-NEXT: [[AND:%.*]] = and i32 [[ADD]], sub (i32 0, i32 zext (i1 icmp eq (i32* getelementptr inbounds ([1 x i32]* @b, i64 0, i64 0), i32* @c) to i32))
; CHECK-NEXT: ret i32 [[AND]]
define i32 @fn1(i32 %a) {
  %xor = add i32 %a, -1
  %mul = mul nsw i32 %xor, zext (i1 icmp eq (i32* getelementptr inbounds ([1 x i32]* @b, i64 0, i64 0), i32* @c) to i32)
  ret i32 %mul
}
