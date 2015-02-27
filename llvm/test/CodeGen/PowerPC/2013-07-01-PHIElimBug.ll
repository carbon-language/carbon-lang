; RUN: llc < %s -verify-machineinstrs | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@g_51 = external global [8 x i32], align 4

; CHECK: func_7

; Function Attrs: nounwind
define fastcc void @func_7() #0 {
entry:
  %arrayidx638 = getelementptr inbounds [3 x [1 x i32]], [3 x [1 x i32]]* undef, i64 0, i64 1, i64 0
  br i1 undef, label %for.cond940, label %if.end1018

for.cond940:                                      ; preds = %for.cond940, %if.else876
  %l_655.1 = phi i32* [ getelementptr inbounds ([8 x i32]* @g_51, i64 0, i64 6), %entry ], [ %l_654.0, %for.cond940 ]
  %l_654.0 = phi i32* [ null, %entry ], [ %arrayidx638, %for.cond940 ]
  %exitcond = icmp eq i32 undef, 20
  br i1 %exitcond, label %if.end1018, label %for.cond940

if.end1018:                                       ; preds = %for.end957, %for.end834
  %l_655.3.ph33 = phi i32* [ %l_655.1, %for.cond940 ], [ getelementptr inbounds ([8 x i32]* @g_51, i64 0, i64 6), %entry ]
  store i32 0, i32* %l_655.3.ph33, align 4
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
