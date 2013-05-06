; Test 32-bit comparisons in which the second operand is a PC-relative
; variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

@g = global i32 1

; Check signed comparisons.
define i32 @f1(i32 %src1) {
; CHECK: f1:
; CHECK: crl %r2, g
; CHECK-NEXT: j{{g?}}l
; CHECK: br %r14
entry:
  %src2 = load i32 *@g
  %cond = icmp slt i32 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i32 %src1, %src1
  br label %exit
exit:
  %res = phi i32 [ %src1, %entry ], [ %mul, %mulb ]
  ret i32 %res
}

; Check unsigned comparisons.
define i32 @f2(i32 %src1) {
; CHECK: f2:
; CHECK: clrl %r2, g
; CHECK-NEXT: j{{g?}}l
; CHECK: br %r14
entry:
  %src2 = load i32 *@g
  %cond = icmp ult i32 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i32 %src1, %src1
  br label %exit
exit:
  %res = phi i32 [ %src1, %entry ], [ %mul, %mulb ]
  ret i32 %res
}

; Check equality, which can use CRL or CLRL.
define i32 @f3(i32 %src1) {
; CHECK: f3:
; CHECK: c{{l?}}rl %r2, g
; CHECK-NEXT: j{{g?}}e
; CHECK: br %r14
entry:
  %src2 = load i32 *@g
  %cond = icmp eq i32 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i32 %src1, %src1
  br label %exit
exit:
  %res = phi i32 [ %src1, %entry ], [ %mul, %mulb ]
  ret i32 %res
}

; ...likewise inequality.
define i32 @f4(i32 %src1) {
; CHECK: f4:
; CHECK: c{{l?}}rl %r2, g
; CHECK-NEXT: j{{g?}}lh
; CHECK: br %r14
entry:
  %src2 = load i32 *@g
  %cond = icmp ne i32 %src1, %src2
  br i1 %cond, label %exit, label %mulb
mulb:
  %mul = mul i32 %src1, %src1
  br label %exit
exit:
  %res = phi i32 [ %src1, %entry ], [ %mul, %mulb ]
  ret i32 %res
}
