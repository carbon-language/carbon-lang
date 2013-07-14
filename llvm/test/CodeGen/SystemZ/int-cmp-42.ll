; Test 64-bit comparisons in which the second operand is zero-extended
; from a PC-relative i32.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

@g = global i32 1
@h = global i32 1, align 2, section "foo"

; Check unsigned comparison.
define i64 @f1(i64 %src1) {
; CHECK-LABEL: f1:
; CHECK: clgfrl %r2, g
; CHECK-NEXT: jl
; CHECK: br %r14
entry:
  %val = load i32 *@g
  %src2 = zext i32 %val to i64
  %cond = icmp ult i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %res = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  ret i64 %res
}

; Check signed comparison.
define i64 @f2(i64 %src1) {
; CHECK-LABEL: f2:
; CHECK-NOT: clgfrl
; CHECK: br %r14
entry:
  %val = load i32 *@g
  %src2 = zext i32 %val to i64
  %cond = icmp slt i64 %src1, %src2
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
; CHECK: clgfrl %r2, g
; CHECK-NEXT: je
; CHECK: br %r14
entry:
  %val = load i32 *@g
  %src2 = zext i32 %val to i64
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
; CHECK: clgfrl %r2, g
; CHECK-NEXT: jlh
; CHECK: br %r14
entry:
  %val = load i32 *@g
  %src2 = zext i32 %val to i64
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
; CHECK: larl [[REG:%r[0-5]]], h
; CHECK: clgf %r2, 0([[REG]])
; CHECK-NEXT: jl
; CHECK: br %r14
entry:
  %val = load i32 *@h, align 2
  %src2 = zext i32 %val to i64
  %cond = icmp ult i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %res = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  ret i64 %res
}
