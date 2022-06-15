; RUN: opt %loadPolly -polly-print-ast -disable-output < %s \
; RUN:   -polly-invariant-load-hoisting \
; RUN:   | FileCheck %s

; CHECK: if (1 && 1 && (&MemRef_X[1] <= &MemRef_BaseA[0] || &MemRef_BaseA[1024] <= &MemRef_X[0]) && (&MemRef_X[1] <= &MemRef_BaseB[0] || &MemRef_BaseB[1024] <= &MemRef_X[0]))

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(float** nocapture readonly %X) {
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.011 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %conv = sitofp i64 %i.011 to float
  %BaseA = load float*, float** %X, align 8
  %BaseB = load float*, float** %X, align 8
  %arrayidx = getelementptr inbounds float, float* %BaseA, i64 %i.011
  %A = load float, float* %arrayidx, align 4
  %add = fadd float %A, %conv
  store float %add, float* %arrayidx, align 4
  %arrayidxB = getelementptr inbounds float, float* %BaseB, i64 %i.011
  %B = load float, float* %arrayidxB, align 4
  %addB = fadd float %B, %conv
  store float %addB, float* %arrayidxB, align 4
  %inc = add nuw nsw i64 %i.011, 1
  %exitcond = icmp eq i64 %inc, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}
