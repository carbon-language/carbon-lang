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


;int foo(double * restrict B,  double * restrict A, int n, int m) {
;  double R=A[1];
;  double G=A[0];
;  for (int i=0; i < 100; i++) {
;    R += 10;
;    G += 10;
;    R *= 4;
;    G *= 4;
;    R += 4;
;    G += 4;
;  }
;  B[0] = G;
;  B[1] = R;
;  return 0;
;}

;CHECK: foo2
;CHECK: load <2 x double>
;CHECK: phi <2 x double>
;CHECK: fmul <2 x double>
;CHECK: store <2 x double>
;CHECK: ret
define i32 @foo2(double* noalias nocapture %B, double* noalias nocapture %A, i32 %n, i32 %m) #0 {
entry:
  %arrayidx = getelementptr inbounds double* %A, i64 1
  %0 = load double* %arrayidx, align 8
  %1 = load double* %A, align 8
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.019 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %G.018 = phi double [ %1, %entry ], [ %add5, %for.body ]
  %R.017 = phi double [ %0, %entry ], [ %add4, %for.body ]
  %add = fadd double %R.017, 1.000000e+01
  %add2 = fadd double %G.018, 1.000000e+01
  %mul = fmul double %add, 4.000000e+00
  %mul3 = fmul double %add2, 4.000000e+00
  %add4 = fadd double %mul, 4.000000e+00
  %add5 = fadd double %mul3, 4.000000e+00
  %inc = add nsw i32 %i.019, 1
  %exitcond = icmp eq i32 %inc, 100
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  store double %add5, double* %B, align 8
  %arrayidx7 = getelementptr inbounds double* %B, i64 1
  store double %add4, double* %arrayidx7, align 8
  ret i32 0
}

