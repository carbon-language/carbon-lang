; RUN: opt %loadPolly -S -polly-prepare < %s | FileCheck %s
; ModuleID = 'single_loop_trivial_phi.ll'
;
; int f(int *A, int N) {
;   int i, sum = 0;
;   for (i = 0; i < N; i++)
;     sum += A[i];
;   return sum;
; }
; ModuleID = 'stack-slots.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @f(i32* %A, i32 %N) #0 {
entry:
  %cmp1 = icmp sgt i32 %N, 0
  br i1 %cmp1, label %for.inc.lr.ph, label %for.end

for.inc.lr.ph:                                    ; preds = %entry
  %0 = zext i32 %N to i64
  br label %for.inc

for.inc:                                          ; preds = %for.inc.lr.ph, %for.inc
  %sum.03 = phi i32 [ 0, %for.inc.lr.ph ], [ %add, %for.inc ]
  %indvars.iv2 = phi i64 [ 0, %for.inc.lr.ph ], [ %indvars.iv.next, %for.inc ]
  %arrayidx = getelementptr i32* %A, i64 %indvars.iv2
  %tmp1 = load i32* %arrayidx, align 4
  %add = add nsw i32 %tmp1, %sum.03
  %indvars.iv.next = add nuw nsw i64 %indvars.iv2, 1
  %exitcond = icmp ne i64 %indvars.iv.next, %0
  br i1 %exitcond, label %for.inc, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.inc
  %add.lcssa = phi i32 [ %add, %for.inc ]
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  %sum.0.lcssa = phi i32 [ %add.lcssa, %for.cond.for.end_crit_edge ], [ 0, %entry ]
  ret i32 %sum.0.lcssa
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

; Verify that only two allocas are created.
; Both are needed for the %sum.0 PHI node and none should be created for the
; %sum.0.lcssa PHI node
; CHECK: alloca
; CHECK: alloca
; CHECK-NOT: alloca
