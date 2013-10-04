; RUN: opt < %s -basicaa -slp-vectorizer -slp-threshold=-100 -instcombine -dce -S -mtriple=i386-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32-S128"



; Make sure we order the operands of commutative operations so that we get
; bigger vectorizable trees.

; CHECK-LABEL: shuffle_operands1
; CHECK:         load <2 x double>
; CHECK:         fadd <2 x double>

define void @shuffle_operands1(double * noalias %from, double * noalias %to,
                               double %v1, double %v2) {
  %from_1 = getelementptr double *%from, i64 1
  %v0_1 = load double * %from
  %v0_2 = load double * %from_1
  %v1_1 = fadd double %v0_1, %v1
  %v1_2 = fadd double %v2, %v0_2
  %to_2 = getelementptr double * %to, i64 1
  store double %v1_1, double *%to
  store double %v1_2, double *%to_2
  ret void
}

; CHECK-LABEL: shuffle_preserve_broadcast
; CHECK: %[[BCAST:[a-z0-9]+]] = insertelement <2 x double> undef, double %v0_1
; CHECK:                      = insertelement <2 x double> %[[BCAST]], double %v0_1
define void @shuffle_preserve_broadcast(double * noalias %from,
                                        double * noalias %to,
                                        double %v1, double %v2) {
entry:
br label %lp

lp:
  %p = phi double [ 1.000000e+00, %lp ], [ 0.000000e+00, %entry ]
  %from_1 = getelementptr double *%from, i64 1
  %v0_1 = load double * %from
  %v0_2 = load double * %from_1
  %v1_1 = fadd double %v0_1, %p
  %v1_2 = fadd double %v0_1, %v0_2
  %to_2 = getelementptr double * %to, i64 1
  store double %v1_1, double *%to
  store double %v1_2, double *%to_2
br i1 undef, label %lp, label %ext

ext:
  ret void
}

; CHECK-LABEL: shuffle_preserve_broadcast2
; CHECK: %[[BCAST:[a-z0-9]+]] = insertelement <2 x double> undef, double %v0_1
; CHECK:                      = insertelement <2 x double> %[[BCAST]], double %v0_1
define void @shuffle_preserve_broadcast2(double * noalias %from,
                                        double * noalias %to,
                                        double %v1, double %v2) {
entry:
br label %lp

lp:
  %p = phi double [ 1.000000e+00, %lp ], [ 0.000000e+00, %entry ]
  %from_1 = getelementptr double *%from, i64 1
  %v0_1 = load double * %from
  %v0_2 = load double * %from_1
  %v1_1 = fadd double %p, %v0_1
  %v1_2 = fadd double %v0_2, %v0_1
  %to_2 = getelementptr double * %to, i64 1
  store double %v1_1, double *%to
  store double %v1_2, double *%to_2
br i1 undef, label %lp, label %ext

ext:
  ret void
}

; CHECK-LABEL: shuffle_preserve_broadcast3
; CHECK: %[[BCAST:[a-z0-9]+]] = insertelement <2 x double> undef, double %v0_1
; CHECK:                      = insertelement <2 x double> %[[BCAST]], double %v0_1
define void @shuffle_preserve_broadcast3(double * noalias %from,
                                        double * noalias %to,
                                        double %v1, double %v2) {
entry:
br label %lp

lp:
  %p = phi double [ 1.000000e+00, %lp ], [ 0.000000e+00, %entry ]
  %from_1 = getelementptr double *%from, i64 1
  %v0_1 = load double * %from
  %v0_2 = load double * %from_1
  %v1_1 = fadd double %p, %v0_1
  %v1_2 = fadd double %v0_1, %v0_2
  %to_2 = getelementptr double * %to, i64 1
  store double %v1_1, double *%to
  store double %v1_2, double *%to_2
br i1 undef, label %lp, label %ext

ext:
  ret void
}


; CHECK-LABEL: shuffle_preserve_broadcast4
; CHECK: %[[BCAST:[a-z0-9]+]] = insertelement <2 x double> undef, double %v0_1
; CHECK:                      = insertelement <2 x double> %[[BCAST]], double %v0_1
define void @shuffle_preserve_broadcast4(double * noalias %from,
                                        double * noalias %to,
                                        double %v1, double %v2) {
entry:
br label %lp

lp:
  %p = phi double [ 1.000000e+00, %lp ], [ 0.000000e+00, %entry ]
  %from_1 = getelementptr double *%from, i64 1
  %v0_1 = load double * %from
  %v0_2 = load double * %from_1
  %v1_1 = fadd double %v0_2, %v0_1
  %v1_2 = fadd double %p, %v0_1
  %to_2 = getelementptr double * %to, i64 1
  store double %v1_1, double *%to
  store double %v1_2, double *%to_2
br i1 undef, label %lp, label %ext

ext:
  ret void
}

; CHECK-LABEL: shuffle_preserve_broadcast5
; CHECK: %[[BCAST:[a-z0-9]+]] = insertelement <2 x double> undef, double %v0_1
; CHECK:                      = insertelement <2 x double> %[[BCAST]], double %v0_1
define void @shuffle_preserve_broadcast5(double * noalias %from,
                                        double * noalias %to,
                                        double %v1, double %v2) {
entry:
br label %lp

lp:
  %p = phi double [ 1.000000e+00, %lp ], [ 0.000000e+00, %entry ]
  %from_1 = getelementptr double *%from, i64 1
  %v0_1 = load double * %from
  %v0_2 = load double * %from_1
  %v1_1 = fadd double %v0_1, %v0_2
  %v1_2 = fadd double %p, %v0_1
  %to_2 = getelementptr double * %to, i64 1
  store double %v1_1, double *%to
  store double %v1_2, double *%to_2
br i1 undef, label %lp, label %ext

ext:
  ret void
}


; CHECK-LABEL: shuffle_preserve_broadcast6
; CHECK: %[[BCAST:[a-z0-9]+]] = insertelement <2 x double> undef, double %v0_1
; CHECK:                      = insertelement <2 x double> %[[BCAST]], double %v0_1
define void @shuffle_preserve_broadcast6(double * noalias %from,
                                        double * noalias %to,
                                        double %v1, double %v2) {
entry:
br label %lp

lp:
  %p = phi double [ 1.000000e+00, %lp ], [ 0.000000e+00, %entry ]
  %from_1 = getelementptr double *%from, i64 1
  %v0_1 = load double * %from
  %v0_2 = load double * %from_1
  %v1_1 = fadd double %v0_1, %v0_2
  %v1_2 = fadd double %v0_1, %p
  %to_2 = getelementptr double * %to, i64 1
  store double %v1_1, double *%to
  store double %v1_2, double *%to_2
br i1 undef, label %lp, label %ext

ext:
  ret void
}

; Make sure we don't scramble operands when we reorder them and destroy
; 'good' source order.

; CHECK-LABEL: good_load_order

; CHECK: %[[V1:[0-9]+]] = load <4 x float>*
; CHECK: %[[V2:[0-9]+]] = insertelement <4 x float> undef, float %1, i32 0
; CHECK: %[[V3:[0-9]+]] = shufflevector <4 x float> %[[V2]], <4 x float> %[[V1]], <4 x i32> <i32 0, i32 4, i32 5, i32 6>
; CHECK:                = fmul <4 x float> %[[V1]], %[[V3]]

@a = common global [32000 x float] zeroinitializer, align 16

define void @good_load_order() {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %0 = load float* getelementptr inbounds ([32000 x float]* @a, i64 0, i64 0), align 16
  br label %for.body3

for.body3:
  %1 = phi float [ %0, %for.cond1.preheader ], [ %10, %for.body3 ]
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body3 ]
  %2 = add nsw i64 %indvars.iv, 1
  %arrayidx = getelementptr inbounds [32000 x float]* @a, i64 0, i64 %2
  %3 = load float* %arrayidx, align 4
  %arrayidx5 = getelementptr inbounds [32000 x float]* @a, i64 0, i64 %indvars.iv
  %mul6 = fmul float %3, %1
  store float %mul6, float* %arrayidx5, align 4
  %4 = add nsw i64 %indvars.iv, 2
  %arrayidx11 = getelementptr inbounds [32000 x float]* @a, i64 0, i64 %4
  %5 = load float* %arrayidx11, align 4
  %mul15 = fmul float %5, %3
  store float %mul15, float* %arrayidx, align 4
  %6 = add nsw i64 %indvars.iv, 3
  %arrayidx21 = getelementptr inbounds [32000 x float]* @a, i64 0, i64 %6
  %7 = load float* %arrayidx21, align 4
  %mul25 = fmul float %7, %5
  store float %mul25, float* %arrayidx11, align 4
  %8 = add nsw i64 %indvars.iv, 4
  %arrayidx31 = getelementptr inbounds [32000 x float]* @a, i64 0, i64 %8
  %9 = load float* %arrayidx31, align 4
  %mul35 = fmul float %9, %7
  store float %mul35, float* %arrayidx21, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 5
  %arrayidx41 = getelementptr inbounds [32000 x float]* @a, i64 0, i64 %indvars.iv.next
  %10 = load float* %arrayidx41, align 4
  %mul45 = fmul float %10, %9
  store float %mul45, float* %arrayidx31, align 4
  %11 = trunc i64 %indvars.iv.next to i32
  %cmp2 = icmp slt i32 %11, 31995
  br i1 %cmp2, label %for.body3, label %for.end

for.end:
  ret void
}
