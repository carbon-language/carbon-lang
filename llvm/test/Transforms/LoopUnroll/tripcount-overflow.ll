; RUN: opt < %s -S -unroll-runtime -unroll-count=2 -loop-unroll | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; When prologue is fully unrolled, the branch on its end is unconditional.
; Unrolling it is illegal if we can't prove that trip-count+1 doesn't overflow,
; like in this example, where it comes from an argument.
;
; This test is based on an example from here:
; http://stackoverflow.com/questions/23838661/why-is-clang-optimizing-this-code-out
;
; CHECK: while.body.prol:
; CHECK: br i1
; CHECK: entry.split:

; Function Attrs: nounwind readnone ssp uwtable
define i32 @foo(i32 %N) #0 {
entry:
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %i = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  %cmp = icmp eq i32 %i, %N
  %inc = add i32 %i, 1
  br i1 %cmp, label %while.end, label %while.body

while.end:                                        ; preds = %while.body
  ret i32 %i
}

attributes #0 = { nounwind readnone ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
