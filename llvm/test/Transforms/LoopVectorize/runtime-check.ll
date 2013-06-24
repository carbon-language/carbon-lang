; RUN: opt < %s  -loop-vectorize -force-vector-unroll=1 -force-vector-width=4 -dce -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Make sure we vectorize this loop:
; int foo(float *a, float *b, int n) {
;   for (int i=0; i<n; ++i)
;     a[i] = b[i] * 3;
; }

;CHECK: for.body.preheader:
;CHECK: br i1 %cmp.zero, label %middle.block, label %vector.memcheck
;CHECK: vector.memcheck:
;CHECK: br i1 %memcheck.conflict, label %middle.block, label %vector.ph
;CHECK: load <4 x float>
define i32 @foo(float* nocapture %a, float* nocapture %b, i32 %n) nounwind uwtable ssp {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float* %b, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4
  %mul = fmul float %0, 3.000000e+00
  %arrayidx2 = getelementptr inbounds float* %a, i64 %indvars.iv
  store float %mul, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret i32 undef
}
