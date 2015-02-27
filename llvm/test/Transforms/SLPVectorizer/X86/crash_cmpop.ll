; RUN: opt < %s -basicaa -slp-vectorizer -S

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

define void @testfunc(float* nocapture %dest, float* nocapture readonly %src) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc1.056 = phi float [ 0.000000e+00, %entry ], [ %add13, %for.body ]
  %s1.055 = phi float [ 0.000000e+00, %entry ], [ %cond.i40, %for.body ]
  %s0.054 = phi float [ 0.000000e+00, %entry ], [ %cond.i44, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %src, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx2 = getelementptr inbounds float, float* %dest, i64 %indvars.iv
  store float %acc1.056, float* %arrayidx2, align 4
  %add = fadd float %s0.054, %0
  %add3 = fadd float %s1.055, %0
  %mul = fmul float %s0.054, 0.000000e+00
  %add4 = fadd float %mul, %add3
  %mul5 = fmul float %s1.055, 0.000000e+00
  %add6 = fadd float %mul5, %add
  %cmp.i = fcmp olt float %add6, 1.000000e+00
  %cond.i = select i1 %cmp.i, float %add6, float 1.000000e+00
  %cmp.i51 = fcmp olt float %cond.i, -1.000000e+00
  %cmp.i49 = fcmp olt float %add4, 1.000000e+00
  %cond.i50 = select i1 %cmp.i49, float %add4, float 1.000000e+00
  %cmp.i47 = fcmp olt float %cond.i50, -1.000000e+00
  %cond.i.op = fmul float %cond.i, 0.000000e+00
  %mul10 = select i1 %cmp.i51, float -0.000000e+00, float %cond.i.op
  %cond.i50.op = fmul float %cond.i50, 0.000000e+00
  %mul11 = select i1 %cmp.i47, float -0.000000e+00, float %cond.i50.op
  %add13 = fadd float %mul10, %mul11

  ; The SLPVectorizer crashed in vectorizeChainsInBlock() because it tried
  ; to access the second operand of the following cmp after the cmp itself
  ; was already vectorized and deleted.
  %cmp.i45 = fcmp olt float %add13, 1.000000e+00

  %cond.i46 = select i1 %cmp.i45, float %add13, float 1.000000e+00
  %cmp.i43 = fcmp olt float %cond.i46, -1.000000e+00
  %cond.i44 = select i1 %cmp.i43, float -1.000000e+00, float %cond.i46
  %cmp.i41 = fcmp olt float %mul11, 1.000000e+00
  %cond.i42 = select i1 %cmp.i41, float %mul11, float 1.000000e+00
  %cmp.i39 = fcmp olt float %cond.i42, -1.000000e+00
  %cond.i40 = select i1 %cmp.i39, float -1.000000e+00, float %cond.i42
  %exitcond = icmp eq i64 %indvars.iv.next, 32
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

