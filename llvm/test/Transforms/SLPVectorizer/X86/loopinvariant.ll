; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

;CHECK-LABEL: @foo(
;CHECK: load <4 x i32>
;CHECK: add nsw <4 x i32>
;CHECK: store <4 x i32>
;CHECK: load <4 x i32>
;CHECK: add nsw <4 x i32>
;CHECK: store <4 x i32>
;CHECK: ret
define i32 @foo(i32* nocapture %A, i32 %n) #0 {
entry:
  %cmp62 = icmp sgt i32 %n, 0
  br i1 %cmp62, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add1 = add nsw i32 %0, %n
  store i32 %add1, i32* %arrayidx, align 4
  %1 = or i64 %indvars.iv, 1
  %arrayidx4 = getelementptr inbounds i32, i32* %A, i64 %1
  %2 = load i32, i32* %arrayidx4, align 4
  %add5 = add nsw i32 %2, %n
  store i32 %add5, i32* %arrayidx4, align 4
  %3 = or i64 %indvars.iv, 2
  %arrayidx8 = getelementptr inbounds i32, i32* %A, i64 %3
  %4 = load i32, i32* %arrayidx8, align 4
  %add9 = add nsw i32 %4, %n
  store i32 %add9, i32* %arrayidx8, align 4
  %5 = or i64 %indvars.iv, 3
  %arrayidx12 = getelementptr inbounds i32, i32* %A, i64 %5
  %6 = load i32, i32* %arrayidx12, align 4
  %add13 = add nsw i32 %6, %n
  store i32 %add13, i32* %arrayidx12, align 4
  %7 = or i64 %indvars.iv, 4
  %arrayidx16 = getelementptr inbounds i32, i32* %A, i64 %7
  %8 = load i32, i32* %arrayidx16, align 4
  %add17 = add nsw i32 %8, %n
  store i32 %add17, i32* %arrayidx16, align 4
  %9 = or i64 %indvars.iv, 5
  %arrayidx20 = getelementptr inbounds i32, i32* %A, i64 %9
  %10 = load i32, i32* %arrayidx20, align 4
  %add21 = add nsw i32 %10, %n
  store i32 %add21, i32* %arrayidx20, align 4
  %11 = or i64 %indvars.iv, 6
  %arrayidx24 = getelementptr inbounds i32, i32* %A, i64 %11
  %12 = load i32, i32* %arrayidx24, align 4
  %add25 = add nsw i32 %12, %n
  store i32 %add25, i32* %arrayidx24, align 4
  %13 = or i64 %indvars.iv, 7
  %arrayidx28 = getelementptr inbounds i32, i32* %A, i64 %13
  %14 = load i32, i32* %arrayidx28, align 4
  %add29 = add nsw i32 %14, %n
  store i32 %add29, i32* %arrayidx28, align 4
  %indvars.iv.next = add i64 %indvars.iv, 8
  %15 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %15, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret i32 undef
}

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
