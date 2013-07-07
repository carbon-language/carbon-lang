; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

define void @main() {
entry:
  br label %for.body

for.body:                                         ; preds = %for.end44, %entry
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %if.then25, %for.body
  br label %for.body6

for.body6:                                        ; preds = %for.inc21, %for.cond4.preheader
  br label %for.body12

for.body12:                                       ; preds = %if.end, %for.body6
  %fZImg.069 = phi double [ undef, %for.body6 ], [ %add19, %if.end ]
  %fZReal.068 = phi double [ undef, %for.body6 ], [ %add20, %if.end ]
  %mul13 = fmul double %fZReal.068, %fZReal.068
  %mul14 = fmul double %fZImg.069, %fZImg.069
  %add15 = fadd double %mul13, %mul14
  %cmp16 = fcmp ogt double %add15, 4.000000e+00
  br i1 %cmp16, label %for.inc21, label %if.end

if.end:                                           ; preds = %for.body12
  %mul18 = fmul double undef, %fZImg.069
  %add19 = fadd double undef, %mul18
  %sub = fsub double %mul13, %mul14
  %add20 = fadd double undef, %sub
  br i1 undef, label %for.body12, label %for.inc21

for.inc21:                                        ; preds = %if.end, %for.body12
  br i1 undef, label %for.end23, label %for.body6

for.end23:                                        ; preds = %for.inc21
  br i1 undef, label %if.then25, label %if.then26

if.then25:                                        ; preds = %for.end23
  br i1 undef, label %for.end44, label %for.cond4.preheader

if.then26:                                        ; preds = %for.end23
  unreachable

for.end44:                                        ; preds = %if.then25
  br i1 undef, label %for.end48, label %for.body

for.end48:                                        ; preds = %for.end44
  ret void
}

