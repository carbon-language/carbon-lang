; RUN: opt %loadPolly -S -polly-prepare < %s | FileCheck %s
; ModuleID = 'multiple_loops_trivial_phis.ll'
;
; int f(int * __restrict__ A) {
;   int i, j, sum = 0;
;   for (i = 0; i < 100; i++) {
;     sum *= 2;
;     for (j = 0; j < 100; j++) {
;       sum += A[i+j];
;     }
;   }
;   return sum;
; }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @f(i32* noalias %A) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc5
  %sum.04 = phi i32 [ 0, %entry ], [ %add4.lcssa, %for.inc5 ]
  %indvars.iv23 = phi i64 [ 0, %entry ], [ %2, %for.inc5 ]
  %mul = shl nsw i32 %sum.04, 1
  br label %for.inc

for.inc:                                          ; preds = %for.body, %for.inc
  %sum.12 = phi i32 [ %mul, %for.body ], [ %add4, %for.inc ]
  %indvars.iv1 = phi i64 [ 0, %for.body ], [ %1, %for.inc ]
  %0 = add i64 %indvars.iv23, %indvars.iv1
  %arrayidx = getelementptr i32* %A, i64 %0
  %tmp5 = load i32* %arrayidx, align 4
  %add4 = add nsw i32 %tmp5, %sum.12
  %1 = add nuw nsw i64 %indvars.iv1, 1
  %exitcond5 = icmp eq i64 %1, 100
  br i1 %exitcond5, label %for.inc5, label %for.inc

for.inc5:                                         ; preds = %for.inc
  %add4.lcssa = phi i32 [ %add4, %for.inc ]
  %2 = add nuw nsw i64 %indvars.iv23, 1
  %exitcond = icmp eq i64 %2, 100
  br i1 %exitcond, label %for.end7, label %for.body

for.end7:                                         ; preds = %for.inc5
  %add4.lcssa.lcssa = phi i32 [ %add4.lcssa, %for.inc5 ]
  ret i32 %add4.lcssa.lcssa
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

; Verify that only two allocas are created. (instead of 4!)
; CHECK: alloca
; CHECK: alloca
; CHECK-NOT: alloca
