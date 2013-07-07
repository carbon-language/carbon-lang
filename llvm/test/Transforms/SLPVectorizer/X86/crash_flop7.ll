; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; Function Attrs: nounwind ssp uwtable
define void @main() #0 {
entry:
  br i1 undef, label %while.body, label %while.end

while.body:                                       ; preds = %entry
  unreachable

while.end:                                        ; preds = %entry
  br i1 undef, label %for.end80, label %for.body75.lr.ph

for.body75.lr.ph:                                 ; preds = %while.end
  br label %for.body75

for.body75:                                       ; preds = %for.body75, %for.body75.lr.ph
  br label %for.body75

for.end80:                                        ; preds = %while.end
  br i1 undef, label %for.end300, label %for.body267.lr.ph

for.body267.lr.ph:                                ; preds = %for.end80
  br label %for.body267

for.body267:                                      ; preds = %for.body267, %for.body267.lr.ph
  %s.71010 = phi double [ 0.000000e+00, %for.body267.lr.ph ], [ %add297, %for.body267 ]
  %mul269 = fmul double undef, undef
  %mul270 = fmul double %mul269, %mul269
  %add282 = fadd double undef, undef
  %mul283 = fmul double %mul269, %add282
  %add293 = fadd double undef, undef
  %mul294 = fmul double %mul270, %add293
  %add295 = fadd double undef, %mul294
  %div296 = fdiv double %mul283, %add295
  %add297 = fadd double %s.71010, %div296
  br i1 undef, label %for.body267, label %for.end300

for.end300:                                       ; preds = %for.body267, %for.end80
  unreachable
}

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
