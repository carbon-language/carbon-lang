; RUN: opt %loadPolly -polly-reschedule=0 -polly-loopfusion-greedy=1 -polly-postopts=0 -polly-opt-isl -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-reschedule=1 -polly-loopfusion-greedy=1 -polly-postopts=0 -polly-opt-isl -analyze < %s | FileCheck %s

; This could theoretically be fused by adjusting the offset of the second loop by %k (instead of relying on schedule dimensions).

define void @func(i32 %n, double* noalias nonnull %A, i32 %k) {
entry:
  br label %for1

for1:
  %j1 = phi i32 [0, %entry], [%j1.inc, %inc1]
  %j1.cmp = icmp slt i32 %j1, %n
  br i1 %j1.cmp, label %body1, label %exit1

    body1:
      %arrayidx1 = getelementptr inbounds double, double* %A, i32 %j1
      store double 21.0, double* %arrayidx1
      br label %inc1

inc1:
  %j1.inc = add nuw nsw i32 %j1, 1
  br label %for1

exit1:
  br label %for2

for2:
  %j2 = phi i32 [0, %exit1], [%j2.inc, %inc2]
  %j2.cmp = icmp slt i32 %j2, %n
  br i1 %j2.cmp, label %body2, label %exit2

    body2:
      %idx2 = add i32 %j2, %k
      %arrayidx2 = getelementptr inbounds double, double* %A, i32 %idx2
      store double 42.0, double* %arrayidx2
      br label %inc2

inc2:
  %j2.inc = add nuw nsw i32 %j2, 1
  br label %for2

exit2:
  br label %return

return:
  ret void
}


; CHECK:      Calculated schedule:
; CHECK-NEXT: n/a
