; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; Function Attrs: nounwind ssp uwtable
define void @RCModelEstimator() {
entry:
  br i1 undef, label %for.body.lr.ph, label %for.end.thread

for.end.thread:                                   ; preds = %entry
  unreachable

for.body.lr.ph:                                   ; preds = %entry
  br i1 undef, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  br i1 undef, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %for.body.lr.ph
  br i1 undef, label %for.body3, label %if.end103

for.cond14.preheader:                             ; preds = %for.inc11
  br i1 undef, label %for.body16.lr.ph, label %if.end103

for.body16.lr.ph:                                 ; preds = %for.cond14.preheader
  br label %for.body16

for.body3:                                        ; preds = %for.inc11, %for.end
  br i1 undef, label %if.then7, label %for.inc11

if.then7:                                         ; preds = %for.body3
  br label %for.inc11

for.inc11:                                        ; preds = %if.then7, %for.body3
  br i1 false, label %for.cond14.preheader, label %for.body3

for.body16:                                       ; preds = %for.body16, %for.body16.lr.ph
  br i1 undef, label %for.end39, label %for.body16

for.end39:                                        ; preds = %for.body16
  br i1 undef, label %if.end103, label %for.cond45.preheader

for.cond45.preheader:                             ; preds = %for.end39
  br i1 undef, label %if.then88, label %if.else

if.then88:                                        ; preds = %for.cond45.preheader
  %mul89 = fmul double 0.000000e+00, 0.000000e+00
  %mul90 = fmul double 0.000000e+00, 0.000000e+00
  %sub91 = fsub double %mul89, %mul90
  %div92 = fdiv double %sub91, undef
  %mul94 = fmul double 0.000000e+00, 0.000000e+00
  %mul95 = fmul double 0.000000e+00, 0.000000e+00
  %sub96 = fsub double %mul94, %mul95
  %div97 = fdiv double %sub96, undef
  br label %if.end103

if.else:                                          ; preds = %for.cond45.preheader
  br label %if.end103

if.end103:                                        ; preds = %if.else, %if.then88, %for.end39, %for.cond14.preheader, %for.end
  %0 = phi double [ 0.000000e+00, %for.end39 ], [ %div97, %if.then88 ], [ 0.000000e+00, %if.else ], [ 0.000000e+00, %for.cond14.preheader ], [ 0.000000e+00, %for.end ]
  %1 = phi double [ undef, %for.end39 ], [ %div92, %if.then88 ], [ undef, %if.else ], [ 0.000000e+00, %for.cond14.preheader ], [ 0.000000e+00, %for.end ]
  ret void
}


define void @intrapred_luma() {
entry:
  %conv153 = trunc i32 undef to i16
  %arrayidx154 = getelementptr inbounds [13 x i16]* undef, i64 0, i64 12
  store i16 %conv153, i16* %arrayidx154, align 8
  %arrayidx155 = getelementptr inbounds [13 x i16]* undef, i64 0, i64 11
  store i16 %conv153, i16* %arrayidx155, align 2
  %arrayidx156 = getelementptr inbounds [13 x i16]* undef, i64 0, i64 10
  store i16 %conv153, i16* %arrayidx156, align 4
  ret void
}

define fastcc void @dct36(double* %inbuf) {
entry:
  %arrayidx41 = getelementptr inbounds double* %inbuf, i64 2
  %arrayidx44 = getelementptr inbounds double* %inbuf, i64 1
  %0 = load double* %arrayidx44, align 8
  %add46 = fadd double %0, undef
  store double %add46, double* %arrayidx41, align 8
  %1 = load double* %inbuf, align 8
  %add49 = fadd double %1, %0
  store double %add49, double* %arrayidx44, align 8
  ret void
}
