; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=i386-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.8.0"

;int test(double *G) {
;  G[0] = 1+G[5]*4;
;  G[1] = 6+G[6]*3;
;  G[2] = 7+G[5]*4;
;  G[3] = 8+G[6]*4;
;}

;CHECK: @test
;CHECK: insertelement <2 x double> undef
;CHECK-NEXT: insertelement <2 x double>
;CHECK-NEXT: fadd <2 x double>
;CHECK: store <2 x double>
;CHECK:  insertelement <2 x double>
;CHECK-NEXT:  fadd <2 x double>
;CHECK:  store <2 x double>
;CHECK: ret i32

define i32 @test(double* nocapture %G) {
entry:
  %arrayidx = getelementptr inbounds double* %G, i64 5
  %0 = load double* %arrayidx, align 8
  %mul = fmul double %0, 4.000000e+00
  %add = fadd double %mul, 1.000000e+00
  store double %add, double* %G, align 8
  %arrayidx2 = getelementptr inbounds double* %G, i64 6
  %1 = load double* %arrayidx2, align 8
  %mul3 = fmul double %1, 3.000000e+00
  %add4 = fadd double %mul3, 6.000000e+00
  %arrayidx5 = getelementptr inbounds double* %G, i64 1
  store double %add4, double* %arrayidx5, align 8
  %add8 = fadd double %mul, 7.000000e+00
  %arrayidx9 = getelementptr inbounds double* %G, i64 2
  store double %add8, double* %arrayidx9, align 8
  %mul11 = fmul double %1, 4.000000e+00
  %add12 = fadd double %mul11, 8.000000e+00
  %arrayidx13 = getelementptr inbounds double* %G, i64 3
  store double %add12, double* %arrayidx13, align 8
  ret i32 undef
}

;int foo(double *A, int n) {
;  A[0] = A[0] * 7.9 * n + 6.0;
;  A[1] = A[1] * 7.7 * n + 2.0;
;  A[2] = A[2] * 7.6 * n + 3.0;
;  A[3] = A[3] * 7.4 * n + 4.0;
;}
;CHECK: @foo
;CHECK: insertelement <2 x double>
;CHECK: insertelement <2 x double>
;CHECK-NOT: insertelement <2 x double>
;CHECK: ret
define i32 @foo(double* nocapture %A, i32 %n) {
entry:
  %0 = load double* %A, align 8
  %mul = fmul double %0, 7.900000e+00
  %conv = sitofp i32 %n to double
  %mul1 = fmul double %conv, %mul
  %add = fadd double %mul1, 6.000000e+00
  store double %add, double* %A, align 8
  %arrayidx3 = getelementptr inbounds double* %A, i64 1
  %1 = load double* %arrayidx3, align 8
  %mul4 = fmul double %1, 7.700000e+00
  %mul6 = fmul double %conv, %mul4
  %add7 = fadd double %mul6, 2.000000e+00
  store double %add7, double* %arrayidx3, align 8
  %arrayidx9 = getelementptr inbounds double* %A, i64 2
  %2 = load double* %arrayidx9, align 8
  %mul10 = fmul double %2, 7.600000e+00
  %mul12 = fmul double %conv, %mul10
  %add13 = fadd double %mul12, 3.000000e+00
  store double %add13, double* %arrayidx9, align 8
  %arrayidx15 = getelementptr inbounds double* %A, i64 3
  %3 = load double* %arrayidx15, align 8
  %mul16 = fmul double %3, 7.400000e+00
  %mul18 = fmul double %conv, %mul16
  %add19 = fadd double %mul18, 4.000000e+00
  store double %add19, double* %arrayidx15, align 8
  ret i32 undef
}

