; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=i386-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.9.0"

;int foo(double *A, int k) {
;  double A0;
;  double A1;
;  if (k) {
;    A0 = 3;
;    A1 = 5;
;  } else {
;    A0 = A[10];
;    A1 = A[11];
;  }
;  A[0] = A0;
;  A[1] = A1;
;}


;CHECK: i32 @foo
;CHECK: load <2 x double>
;CHECK: phi <2 x double>
;CHECK: store <2 x double>
;CHECK: ret i32 undef
define i32 @foo(double* nocapture %A, i32 %k) {
entry:
  %tobool = icmp eq i32 %k, 0
  br i1 %tobool, label %if.else, label %if.end

if.else:                                          ; preds = %entry
  %arrayidx = getelementptr inbounds double* %A, i64 10
  %0 = load double* %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds double* %A, i64 11
  %1 = load double* %arrayidx1, align 8
  br label %if.end

if.end:                                           ; preds = %entry, %if.else
  %A0.0 = phi double [ %0, %if.else ], [ 3.000000e+00, %entry ]
  %A1.0 = phi double [ %1, %if.else ], [ 5.000000e+00, %entry ]
  store double %A0.0, double* %A, align 8
  %arrayidx3 = getelementptr inbounds double* %A, i64 1
  store double %A1.0, double* %arrayidx3, align 8
  ret i32 undef
}

