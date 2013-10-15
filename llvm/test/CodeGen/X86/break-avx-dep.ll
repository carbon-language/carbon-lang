; RUN: llc < %s -march=x86-64 -mattr=+avx | FileCheck %s
;
; rdar:15221834 False AVX register dependencies cause 5x slowdown on
; flops-6. Make sure the unused register read by vcvtsi2sdq is zeroed
; to avoid cyclic dependence on a write to the same register in a
; previous iteration.

; CHECK-LABEL: t1:
; CHECK-LABEL: %loop
; CHECK: vxorps %[[REG:xmm.]], %{{xmm.}}, %{{xmm.}}
; CHECK: vcvtsi2sdq %{{r[0-9a-x]+}}, %[[REG]], %{{xmm.}}
define i64 @t1(i64* nocapture %x, double* nocapture %y) nounwind {
entry:
  %vx = load i64* %x
  br label %loop
loop:
  %i = phi i64 [ 1, %entry ], [ %inc, %loop ]
  %s1 = phi i64 [ %vx, %entry ], [ %s2, %loop ]
  %fi = sitofp i64 %i to double
  %vy = load double* %y
  %fipy = fadd double %fi, %vy
  %iipy = fptosi double %fipy to i64
  %s2 = add i64 %s1, %iipy
  %inc = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %inc, 156250000
  br i1 %exitcond, label %ret, label %loop
ret:
  ret i64 %s2
}
