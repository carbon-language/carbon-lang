; Test 32-bit comparisons in which the second operand is zero-extended
; from a PC-relative i16.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

@g = global i16 1
@h = global i16 1, align 1, section "foo"

; Check unsigned comparison.
define i32 @f1(i32 %src1) {
; CHECK-LABEL: f1:
; CHECK: clhrl %r2, g
; CHECK-NEXT: jl
; CHECK: br %r14
entry:
  %val = load i16 *@g
  %src2 = zext i16 %val to i32
  %cond = icmp ult i32 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i32 %src1, %src1
  br label %exit
exit:
  %res = phi i32 [ %src1, %entry ], [ %mul, %mulb ]
  ret i32 %res
}

; Check signed comparison.
define i32 @f2(i32 %src1) {
; CHECK-LABEL: f2:
; CHECK-NOT: clhrl
; CHECK: br %r14
entry:
  %val = load i16 *@g
  %src2 = zext i16 %val to i32
  %cond = icmp slt i32 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i32 %src1, %src1
  br label %exit
exit:
  %res = phi i32 [ %src1, %entry ], [ %mul, %mulb ]
  ret i32 %res
}

; Check equality.
define i32 @f3(i32 %src1) {
; CHECK-LABEL: f3:
; CHECK: clhrl %r2, g
; CHECK-NEXT: je
; CHECK: br %r14
entry:
  %val = load i16 *@g
  %src2 = zext i16 %val to i32
  %cond = icmp eq i32 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i32 %src1, %src1
  br label %exit
exit:
  %res = phi i32 [ %src1, %entry ], [ %mul, %mulb ]
  ret i32 %res
}

; Check inequality.
define i32 @f4(i32 %src1) {
; CHECK-LABEL: f4:
; CHECK: clhrl %r2, g
; CHECK-NEXT: jlh
; CHECK: br %r14
entry:
  %val = load i16 *@g
  %src2 = zext i16 %val to i32
  %cond = icmp ne i32 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i32 %src1, %src1
  br label %exit
exit:
  %res = phi i32 [ %src1, %entry ], [ %mul, %mulb ]
  ret i32 %res
}

; Repeat f1 with an unaligned address.
define i32 @f5(i32 %src1) {
; CHECK-LABEL: f5:
; CHECK: lgrl [[REG:%r[0-5]]], h@GOT
; CHECK: llh [[VAL:%r[0-5]]], 0([[REG]])
; CHECK: clr %r2, [[VAL]]
; CHECK-NEXT: jl
; CHECK: br %r14
entry:
  %val = load i16 *@h, align 1
  %src2 = zext i16 %val to i32
  %cond = icmp ult i32 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i32 %src1, %src1
  br label %exit
exit:
  %res = phi i32 [ %src1, %entry ], [ %mul, %mulb ]
  ret i32 %res
}
