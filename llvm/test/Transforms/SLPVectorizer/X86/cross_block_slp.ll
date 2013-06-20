; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; int foo(double *A, float *B, int g) {
;   float B0 = B[0];
;   float B1 = B[1]; <----- BasicBlock #1
;   B0 += 5;
;   B1 += 8;
;
;   if (g) bar();
;
;   A[0] += B0;     <------- BasicBlock #3
;   A[1] += B1;
; }


;CHECK: @foo
;CHECK: load <2 x float>
;CHECK: fadd <2 x float>
;CHECK: call i32
;CHECK: load <2 x double>
;CHECK: fadd <2 x double>
;CHECK: store <2 x double>
;CHECK: ret
define i32 @foo(double* nocapture %A, float* nocapture %B, i32 %g) {
entry:
  %0 = load float* %B, align 4
  %arrayidx1 = getelementptr inbounds float* %B, i64 1
  %1 = load float* %arrayidx1, align 4
  %add = fadd float %0, 5.000000e+00
  %add2 = fadd float %1, 8.000000e+00
  %tobool = icmp eq i32 %g, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:
  %call = tail call i32 (...)* @bar()
  br label %if.end

if.end:
  %conv = fpext float %add to double
  %2 = load double* %A, align 8
  %add4 = fadd double %conv, %2
  store double %add4, double* %A, align 8
  %conv5 = fpext float %add2 to double
  %arrayidx6 = getelementptr inbounds double* %A, i64 1
  %3 = load double* %arrayidx6, align 8
  %add7 = fadd double %conv5, %3
  store double %add7, double* %arrayidx6, align 8
  ret i32 undef
}

declare i32 @bar(...)
