; RUN: opt %loadPolly -polly-region-simplify -polly-codegen-isl -polly-codegen-scev < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @kernel_cholesky(double* %p, [128 x double]* %A) #0 {
entry:
  br i1 undef, label %for.body, label %for.end57

for.body:                                         ; preds = %for.inc55, %entry
  %i.070 = phi i32 [ %inc56, %for.inc55 ], [ 0, %entry ]
  br i1 undef, label %for.body22, label %for.inc55

for.body22:                                       ; preds = %for.end44, %for.body
  %sub28 = add nsw i32 %i.070, -1
  %cmp2961 = icmp slt i32 %sub28, 0
  %idxprom45 = sext i32 %i.070 to i64
  br i1 %cmp2961, label %for.end44, label %for.inc42

for.inc42:                                        ; preds = %for.inc42, %for.body22
  %k.062 = phi i32 [ %inc43, %for.inc42 ], [ 0, %for.body22 ]
  %inc43 = add nsw i32 %k.062, 1
  %cmp29 = icmp sgt i32 %inc43, %sub28
  br i1 %cmp29, label %for.end44, label %for.inc42

for.end44:                                        ; preds = %for.inc42, %for.body22
  %arrayidx46 = getelementptr inbounds double* %p, i64 %idxprom45
  %0 = load double* %arrayidx46, align 8
  %mul47 = fmul double undef, %0
  %arrayidx51 = getelementptr inbounds [128 x double]* %A, i64 undef, i64 undef
  store double %mul47, double* %arrayidx51, align 8
  br i1 undef, label %for.body22, label %for.inc55

for.inc55:                                        ; preds = %for.end44, %for.body
  %inc56 = add nsw i32 %i.070, 1
  br i1 undef, label %for.body, label %for.end57

for.end57:                                        ; preds = %for.inc55, %entry
  ret void
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
