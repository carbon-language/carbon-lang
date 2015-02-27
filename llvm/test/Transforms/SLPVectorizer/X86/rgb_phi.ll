; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=i386-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.9.0"

; We disable the vectorization of <3 x float> for now

; float foo(float *A) {
;
;   float R = A[0];
;   float G = A[1];
;   float B = A[2];
;   for (int i=0; i < 121; i+=3) {
;     R+=A[i+0]*7;
;     G+=A[i+1]*8;
;     B+=A[i+2]*9;
;   }
;
;   return R+G+B;
; }

;CHECK-LABEL: @foo(
;CHECK: br
;CHECK-NOT: phi <3 x float>
;CHECK-NOT: fmul <3 x float>
;CHECK-NOT: fadd <3 x float>
; At the moment we don't sink extractelements.
;CHECK: br
;CHECK-NOT: extractelement
;CHECK-NOT: extractelement
;CHECK-NOT: extractelement
;CHECK: ret

define float @foo(float* nocapture readonly %A) {
entry:
  %0 = load float, float* %A, align 4
  %arrayidx1 = getelementptr inbounds float, float* %A, i64 1
  %1 = load float, float* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds float, float* %A, i64 2
  %2 = load float, float* %arrayidx2, align 4
  br label %for.body

for.body:                                         ; preds = %for.body.for.body_crit_edge, %entry
  %3 = phi float [ %0, %entry ], [ %.pre, %for.body.for.body_crit_edge ]
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body.for.body_crit_edge ]
  %B.032 = phi float [ %2, %entry ], [ %add14, %for.body.for.body_crit_edge ]
  %G.031 = phi float [ %1, %entry ], [ %add9, %for.body.for.body_crit_edge ]
  %R.030 = phi float [ %0, %entry ], [ %add4, %for.body.for.body_crit_edge ]
  %mul = fmul float %3, 7.000000e+00
  %add4 = fadd float %R.030, %mul
  %4 = add nsw i64 %indvars.iv, 1
  %arrayidx7 = getelementptr inbounds float, float* %A, i64 %4
  %5 = load float, float* %arrayidx7, align 4
  %mul8 = fmul float %5, 8.000000e+00
  %add9 = fadd float %G.031, %mul8
  %6 = add nsw i64 %indvars.iv, 2
  %arrayidx12 = getelementptr inbounds float, float* %A, i64 %6
  %7 = load float, float* %arrayidx12, align 4
  %mul13 = fmul float %7, 9.000000e+00
  %add14 = fadd float %B.032, %mul13
  %indvars.iv.next = add i64 %indvars.iv, 3
  %8 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %8, 121
  br i1 %cmp, label %for.body.for.body_crit_edge, label %for.end

for.body.for.body_crit_edge:                      ; preds = %for.body
  %arrayidx3.phi.trans.insert = getelementptr inbounds float, float* %A, i64 %indvars.iv.next
  %.pre = load float, float* %arrayidx3.phi.trans.insert, align 4
  br label %for.body

for.end:                                          ; preds = %for.body
  %add16 = fadd float %add4, %add9
  %add17 = fadd float %add16, %add14
  ret float %add17
}

