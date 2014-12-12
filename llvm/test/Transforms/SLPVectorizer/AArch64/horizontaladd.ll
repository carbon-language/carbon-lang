; RUN: opt < %s -basicaa -slp-vectorizer -S -mtriple=aarch64-unknown-linux-gnu -mcpu=cortex-a57 | FileCheck %s
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; float hadd (float *a) {
;   return (a[0] + a[1]) + (a[2] + a[3]);
; }

; CHECK_LABEL: @hadd
; CHECK: load <2 x float>*
; CHECK: fadd <2 x float>
; CHECK: extractelement <2 x float>
 
define float @hadd(float* nocapture readonly %a) {
entry:
  %0 = load float* %a, align 4
  %arrayidx1 = getelementptr inbounds float* %a, i64 1
  %1 = load float* %arrayidx1, align 4
  %add = fadd float %0, %1
  %arrayidx2 = getelementptr inbounds float* %a, i64 2
  %2 = load float* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds float* %a, i64 3
  %3 = load float* %arrayidx3, align 4
  %add4 = fadd float %2, %3
  %add5 = fadd float %add, %add4
  ret float %add5
}
