; Test 64-bit comparisons in which the second operand is a PC-relative
; variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

@g = global i64 1

; Check signed comparisons.
define i64 @f1(i64 %src1) {
; CHECK: f1:
; CHECK: cgrl %r2, g
; CHECK-NEXT: j{{g?}}l
; CHECK: br %r14
entry:
  %src2 = load i64 *@g
  %cond = icmp slt i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %res = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  ret i64 %res
}

; Check unsigned comparisons.
define i64 @f2(i64 %src1) {
; CHECK: f2:
; CHECK: clgrl %r2, g
; CHECK-NEXT: j{{g?}}l
; CHECK: br %r14
entry:
  %src2 = load i64 *@g
  %cond = icmp ult i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %res = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  ret i64 %res
}

; Check equality, which can use CRL or CLRL.
define i64 @f3(i64 %src1) {
; CHECK: f3:
; CHECK: c{{l?}}grl %r2, g
; CHECK-NEXT: j{{g?}}e
; CHECK: br %r14
entry:
  %src2 = load i64 *@g
  %cond = icmp eq i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %res = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  ret i64 %res
}

; ...likewise inequality.
define i64 @f4(i64 %src1) {
; CHECK: f4:
; CHECK: c{{l?}}grl %r2, g
; CHECK-NEXT: j{{g?}}lh
; CHECK: br %r14
entry:
  %src2 = load i64 *@g
  %cond = icmp ne i64 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i64 %src1, %src1
  br label %exit
exit:
  %res = phi i64 [ %src1, %entry ], [ %mul, %mulb ]
  ret i64 %res
}
