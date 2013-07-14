; Test 64-bit comparisons in which the second operand is sign-extended
; from a PC-relative i16.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

@g = global i16 1
@h = global i16 1, align 1, section "foo"

; Check signed comparison.
define i64 @f1(i64 %src1) {
; CHECK-LABEL: f1:
; CHECK: cghrl %r2, g
; CHECK-NEXT: jl
; CHECK: br %r14
entry:
  %val = load i16 *@g
  %src2 = sext i16 %val to i64
  %cond = icmp slt i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %res = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  ret i64 %res
}

; Check unsigned comparison, which cannot use CHRL.
define i64 @f2(i64 %src1) {
; CHECK-LABEL: f2:
; CHECK-NOT: cghrl
; CHECK: br %r14
entry:
  %val = load i16 *@g
  %src2 = sext i16 %val to i64
  %cond = icmp ult i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %res = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  ret i64 %res
}

; Check equality.
define i64 @f3(i64 %src1) {
; CHECK-LABEL: f3:
; CHECK: cghrl %r2, g
; CHECK-NEXT: je
; CHECK: br %r14
entry:
  %val = load i16 *@g
  %src2 = sext i16 %val to i64
  %cond = icmp eq i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %res = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  ret i64 %res
}

; Check inequality.
define i64 @f4(i64 %src1) {
; CHECK-LABEL: f4:
; CHECK: cghrl %r2, g
; CHECK-NEXT: jlh
; CHECK: br %r14
entry:
  %val = load i16 *@g
  %src2 = sext i16 %val to i64
  %cond = icmp ne i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %res = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  ret i64 %res
}

; Repeat f1 with an unaligned address.
define i64 @f5(i64 %src1) {
; CHECK-LABEL: f5:
; CHECK: lgrl [[REG:%r[0-5]]], h@GOT
; CHECK: cgh %r2, 0([[REG]])
; CHECK-NEXT: jl
; CHECK: br %r14
entry:
  %val = load i16 *@h, align 1
  %src2 = sext i16 %val to i64
  %cond = icmp slt i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %res = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  ret i64 %res
}
