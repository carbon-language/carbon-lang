; RUN: opt < %s -S -loop-unroll -mtriple aarch64 -mcpu=cortex-a57 | FileCheck %s

; Tests for unrolling loops with run-time trip counts

; CHECK:  %xtraiter = and i32 %n
; CHECK:  %lcmp.mod = icmp ne i32 %xtraiter, 0
; CHECK:  br i1 %lcmp.mod, label %for.body.prol, label %for.body.preheader.split

; CHECK:  for.body.prol:
; CHECK:  for.body:

define i32 @test(i32* nocapture %a, i32 %n) nounwind uwtable readonly {
entry:
  %cmp1 = icmp eq i32 %n, 0
  br i1 %cmp1, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %sum.02 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %sum.02
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %sum.0.lcssa
}


