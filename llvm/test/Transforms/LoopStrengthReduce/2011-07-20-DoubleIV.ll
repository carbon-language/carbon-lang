; RUN: opt < %s -loop-reduce -S | FileCheck %s
;
; Test LSR's OptimizeShadowIV. Handle a floating-point IV with a
; nonzero initial value.
; rdar://9786536

; First, make sure LSR doesn't crash on an empty IVUsers list.
; CHECK: @dummyIV
; CHECK-NOT: phi
; CHECK-NOT: sitofp
; CHECK: br
define void @dummyIV() nounwind {
entry:
  br label %loop

loop:
  %i.01 = phi i32 [ -39, %entry ], [ %inc, %loop ]
  %conv7 = sitofp i32 %i.01 to double
  %inc = add nsw i32 %i.01, 1
  br i1 undef, label %loop, label %for.end

for.end:
  unreachable
}
