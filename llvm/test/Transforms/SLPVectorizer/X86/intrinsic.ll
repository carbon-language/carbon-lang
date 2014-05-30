; RUN: opt < %s -basicaa -slp-vectorizer -slp-threshold=-999 -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

declare double @llvm.fabs.f64(double) nounwind readnone

;CHECK-LABEL: @vec_fabs_f64(
;CHECK: load <2 x double>
;CHECK: load <2 x double>
;CHECK: call <2 x double> @llvm.fabs.v2f64
;CHECK: store <2 x double>
;CHECK: ret
define void @vec_fabs_f64(double* %a, double* %b, double* %c) {
entry:
  %i0 = load double* %a, align 8
  %i1 = load double* %b, align 8
  %mul = fmul double %i0, %i1
  %call = tail call double @llvm.fabs.f64(double %mul) nounwind readnone
  %arrayidx3 = getelementptr inbounds double* %a, i64 1
  %i3 = load double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double* %b, i64 1
  %i4 = load double* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  %call5 = tail call double @llvm.fabs.f64(double %mul5) nounwind readnone
  store double %call, double* %c, align 8
  %arrayidx5 = getelementptr inbounds double* %c, i64 1
  store double %call5, double* %arrayidx5, align 8
  ret void
}

declare float @llvm.copysign.f32(float, float) nounwind readnone

;CHECK-LABEL: @vec_copysign_f32(
;CHECK: load <4 x float>
;CHECK: load <4 x float>
;CHECK: call <4 x float> @llvm.copysign.v4f32
;CHECK: store <4 x float>
;CHECK: ret
define void @vec_copysign_f32(float* %a, float* %b, float* noalias %c) {
entry:
  %0 = load float* %a, align 4
  %1 = load float* %b, align 4
  %call0 = tail call float @llvm.copysign.f32(float %0, float %1) nounwind readnone
  store float %call0, float* %c, align 4

  %ix2 = getelementptr inbounds float* %a, i64 1
  %2 = load float* %ix2, align 4
  %ix3 = getelementptr inbounds float* %b, i64 1
  %3 = load float* %ix3, align 4
  %call1 = tail call float @llvm.copysign.f32(float %2, float %3) nounwind readnone
  %c1 = getelementptr inbounds float* %c, i64 1
  store float %call1, float* %c1, align 4

  %ix4 = getelementptr inbounds float* %a, i64 2
  %4 = load float* %ix4, align 4
  %ix5 = getelementptr inbounds float* %b, i64 2
  %5 = load float* %ix5, align 4
  %call2 = tail call float @llvm.copysign.f32(float %4, float %5) nounwind readnone
  %c2 = getelementptr inbounds float* %c, i64 2
  store float %call2, float* %c2, align 4

  %ix6 = getelementptr inbounds float* %a, i64 3
  %6 = load float* %ix6, align 4
  %ix7 = getelementptr inbounds float* %b, i64 3
  %7 = load float* %ix7, align 4
  %call3 = tail call float @llvm.copysign.f32(float %6, float %7) nounwind readnone
  %c3 = getelementptr inbounds float* %c, i64 3
  store float %call3, float* %c3, align 4

  ret void
}

declare i32 @llvm.bswap.i32(i32) nounwind readnone

define void @vec_bswap_i32(i32* %a, i32* %b, i32* %c) {
entry:
  %i0 = load i32* %a, align 4
  %i1 = load i32* %b, align 4
  %add1 = add i32 %i0, %i1
  %call1 = tail call i32 @llvm.bswap.i32(i32 %add1) nounwind readnone

  %arrayidx2 = getelementptr inbounds i32* %a, i32 1
  %i2 = load i32* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32* %b, i32 1
  %i3 = load i32* %arrayidx3, align 4
  %add2 = add i32 %i2, %i3
  %call2 = tail call i32 @llvm.bswap.i32(i32 %add2) nounwind readnone

  %arrayidx4 = getelementptr inbounds i32* %a, i32 2
  %i4 = load i32* %arrayidx4, align 4
  %arrayidx5 = getelementptr inbounds i32* %b, i32 2
  %i5 = load i32* %arrayidx5, align 4
  %add3 = add i32 %i4, %i5
  %call3 = tail call i32 @llvm.bswap.i32(i32 %add3) nounwind readnone

  %arrayidx6 = getelementptr inbounds i32* %a, i32 3
  %i6 = load i32* %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds i32* %b, i32 3
  %i7 = load i32* %arrayidx7, align 4
  %add4 = add i32 %i6, %i7
  %call4 = tail call i32 @llvm.bswap.i32(i32 %add4) nounwind readnone

  store i32 %call1, i32* %c, align 4
  %arrayidx8 = getelementptr inbounds i32* %c, i32 1
  store i32 %call2, i32* %arrayidx8, align 4
  %arrayidx9 = getelementptr inbounds i32* %c, i32 2
  store i32 %call3, i32* %arrayidx9, align 4
  %arrayidx10 = getelementptr inbounds i32* %c, i32 3
  store i32 %call4, i32* %arrayidx10, align 4
  ret void

; CHECK-LABEL: @vec_bswap_i32(
; CHECK: load <4 x i32>
; CHECK: load <4 x i32>
; CHECK: call <4 x i32> @llvm.bswap.v4i32
; CHECK: store <4 x i32>
; CHECK: ret
}

declare i32 @llvm.ctlz.i32(i32,i1) nounwind readnone

define void @vec_ctlz_i32(i32* %a, i32* %b, i32* %c, i1) {
entry:
  %i0 = load i32* %a, align 4
  %i1 = load i32* %b, align 4
  %add1 = add i32 %i0, %i1
  %call1 = tail call i32 @llvm.ctlz.i32(i32 %add1,i1 true) nounwind readnone

  %arrayidx2 = getelementptr inbounds i32* %a, i32 1
  %i2 = load i32* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32* %b, i32 1
  %i3 = load i32* %arrayidx3, align 4
  %add2 = add i32 %i2, %i3
  %call2 = tail call i32 @llvm.ctlz.i32(i32 %add2,i1 true) nounwind readnone

  %arrayidx4 = getelementptr inbounds i32* %a, i32 2
  %i4 = load i32* %arrayidx4, align 4
  %arrayidx5 = getelementptr inbounds i32* %b, i32 2
  %i5 = load i32* %arrayidx5, align 4
  %add3 = add i32 %i4, %i5
  %call3 = tail call i32 @llvm.ctlz.i32(i32 %add3,i1 true) nounwind readnone

  %arrayidx6 = getelementptr inbounds i32* %a, i32 3
  %i6 = load i32* %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds i32* %b, i32 3
  %i7 = load i32* %arrayidx7, align 4
  %add4 = add i32 %i6, %i7
  %call4 = tail call i32 @llvm.ctlz.i32(i32 %add4,i1 true) nounwind readnone

  store i32 %call1, i32* %c, align 4
  %arrayidx8 = getelementptr inbounds i32* %c, i32 1
  store i32 %call2, i32* %arrayidx8, align 4
  %arrayidx9 = getelementptr inbounds i32* %c, i32 2
  store i32 %call3, i32* %arrayidx9, align 4
  %arrayidx10 = getelementptr inbounds i32* %c, i32 3
  store i32 %call4, i32* %arrayidx10, align 4
  ret void

; CHECK-LABEL: @vec_ctlz_i32(
; CHECK: load <4 x i32>
; CHECK: load <4 x i32>
; CHECK: call <4 x i32> @llvm.ctlz.v4i32
; CHECK: store <4 x i32>
; CHECK: ret
}

define void @vec_ctlz_i32_neg(i32* %a, i32* %b, i32* %c, i1) {
entry:
  %i0 = load i32* %a, align 4
  %i1 = load i32* %b, align 4
  %add1 = add i32 %i0, %i1
  %call1 = tail call i32 @llvm.ctlz.i32(i32 %add1,i1 true) nounwind readnone

  %arrayidx2 = getelementptr inbounds i32* %a, i32 1
  %i2 = load i32* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32* %b, i32 1
  %i3 = load i32* %arrayidx3, align 4
  %add2 = add i32 %i2, %i3
  %call2 = tail call i32 @llvm.ctlz.i32(i32 %add2,i1 false) nounwind readnone

  %arrayidx4 = getelementptr inbounds i32* %a, i32 2
  %i4 = load i32* %arrayidx4, align 4
  %arrayidx5 = getelementptr inbounds i32* %b, i32 2
  %i5 = load i32* %arrayidx5, align 4
  %add3 = add i32 %i4, %i5
  %call3 = tail call i32 @llvm.ctlz.i32(i32 %add3,i1 true) nounwind readnone

  %arrayidx6 = getelementptr inbounds i32* %a, i32 3
  %i6 = load i32* %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds i32* %b, i32 3
  %i7 = load i32* %arrayidx7, align 4
  %add4 = add i32 %i6, %i7
  %call4 = tail call i32 @llvm.ctlz.i32(i32 %add4,i1 false) nounwind readnone

  store i32 %call1, i32* %c, align 4
  %arrayidx8 = getelementptr inbounds i32* %c, i32 1
  store i32 %call2, i32* %arrayidx8, align 4
  %arrayidx9 = getelementptr inbounds i32* %c, i32 2
  store i32 %call3, i32* %arrayidx9, align 4
  %arrayidx10 = getelementptr inbounds i32* %c, i32 3
  store i32 %call4, i32* %arrayidx10, align 4
  ret void

; CHECK-LABEL: @vec_ctlz_i32_neg(
; CHECK-NOT: call <4 x i32> @llvm.ctlz.v4i32

}


declare i32 @llvm.cttz.i32(i32,i1) nounwind readnone

define void @vec_cttz_i32(i32* %a, i32* %b, i32* %c, i1) {
entry:
  %i0 = load i32* %a, align 4
  %i1 = load i32* %b, align 4
  %add1 = add i32 %i0, %i1
  %call1 = tail call i32 @llvm.cttz.i32(i32 %add1,i1 true) nounwind readnone

  %arrayidx2 = getelementptr inbounds i32* %a, i32 1
  %i2 = load i32* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32* %b, i32 1
  %i3 = load i32* %arrayidx3, align 4
  %add2 = add i32 %i2, %i3
  %call2 = tail call i32 @llvm.cttz.i32(i32 %add2,i1 true) nounwind readnone

  %arrayidx4 = getelementptr inbounds i32* %a, i32 2
  %i4 = load i32* %arrayidx4, align 4
  %arrayidx5 = getelementptr inbounds i32* %b, i32 2
  %i5 = load i32* %arrayidx5, align 4
  %add3 = add i32 %i4, %i5
  %call3 = tail call i32 @llvm.cttz.i32(i32 %add3,i1 true) nounwind readnone

  %arrayidx6 = getelementptr inbounds i32* %a, i32 3
  %i6 = load i32* %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds i32* %b, i32 3
  %i7 = load i32* %arrayidx7, align 4
  %add4 = add i32 %i6, %i7
  %call4 = tail call i32 @llvm.cttz.i32(i32 %add4,i1 true) nounwind readnone

  store i32 %call1, i32* %c, align 4
  %arrayidx8 = getelementptr inbounds i32* %c, i32 1
  store i32 %call2, i32* %arrayidx8, align 4
  %arrayidx9 = getelementptr inbounds i32* %c, i32 2
  store i32 %call3, i32* %arrayidx9, align 4
  %arrayidx10 = getelementptr inbounds i32* %c, i32 3
  store i32 %call4, i32* %arrayidx10, align 4
  ret void

; CHECK-LABEL: @vec_cttz_i32(
; CHECK: load <4 x i32>
; CHECK: load <4 x i32>
; CHECK: call <4 x i32> @llvm.cttz.v4i32
; CHECK: store <4 x i32>
; CHECK: ret
}

define void @vec_cttz_i32_neg(i32* %a, i32* %b, i32* %c, i1) {
entry:
  %i0 = load i32* %a, align 4
  %i1 = load i32* %b, align 4
  %add1 = add i32 %i0, %i1
  %call1 = tail call i32 @llvm.cttz.i32(i32 %add1,i1 true) nounwind readnone

  %arrayidx2 = getelementptr inbounds i32* %a, i32 1
  %i2 = load i32* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32* %b, i32 1
  %i3 = load i32* %arrayidx3, align 4
  %add2 = add i32 %i2, %i3
  %call2 = tail call i32 @llvm.cttz.i32(i32 %add2,i1 false) nounwind readnone

  %arrayidx4 = getelementptr inbounds i32* %a, i32 2
  %i4 = load i32* %arrayidx4, align 4
  %arrayidx5 = getelementptr inbounds i32* %b, i32 2
  %i5 = load i32* %arrayidx5, align 4
  %add3 = add i32 %i4, %i5
  %call3 = tail call i32 @llvm.cttz.i32(i32 %add3,i1 true) nounwind readnone

  %arrayidx6 = getelementptr inbounds i32* %a, i32 3
  %i6 = load i32* %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds i32* %b, i32 3
  %i7 = load i32* %arrayidx7, align 4
  %add4 = add i32 %i6, %i7
  %call4 = tail call i32 @llvm.cttz.i32(i32 %add4,i1 false) nounwind readnone

  store i32 %call1, i32* %c, align 4
  %arrayidx8 = getelementptr inbounds i32* %c, i32 1
  store i32 %call2, i32* %arrayidx8, align 4
  %arrayidx9 = getelementptr inbounds i32* %c, i32 2
  store i32 %call3, i32* %arrayidx9, align 4
  %arrayidx10 = getelementptr inbounds i32* %c, i32 3
  store i32 %call4, i32* %arrayidx10, align 4
  ret void

; CHECK-LABEL: @vec_cttz_i32_neg(
; CHECK-NOT: call <4 x i32> @llvm.cttz.v4i32
}


declare float @llvm.powi.f32(float, i32)
define void @vec_powi_f32(float* %a, float* %b, float* %c, i32 %P) {
entry:
  %i0 = load float* %a, align 4
  %i1 = load float* %b, align 4
  %add1 = fadd float %i0, %i1
  %call1 = tail call float @llvm.powi.f32(float %add1,i32 %P) nounwind readnone

  %arrayidx2 = getelementptr inbounds float* %a, i32 1
  %i2 = load float* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds float* %b, i32 1
  %i3 = load float* %arrayidx3, align 4
  %add2 = fadd float %i2, %i3
  %call2 = tail call float @llvm.powi.f32(float %add2,i32 %P) nounwind readnone

  %arrayidx4 = getelementptr inbounds float* %a, i32 2
  %i4 = load float* %arrayidx4, align 4
  %arrayidx5 = getelementptr inbounds float* %b, i32 2
  %i5 = load float* %arrayidx5, align 4
  %add3 = fadd float %i4, %i5
  %call3 = tail call float @llvm.powi.f32(float %add3,i32 %P) nounwind readnone

  %arrayidx6 = getelementptr inbounds float* %a, i32 3
  %i6 = load float* %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds float* %b, i32 3
  %i7 = load float* %arrayidx7, align 4
  %add4 = fadd float %i6, %i7
  %call4 = tail call float @llvm.powi.f32(float %add4,i32 %P) nounwind readnone

  store float %call1, float* %c, align 4
  %arrayidx8 = getelementptr inbounds float* %c, i32 1
  store float %call2, float* %arrayidx8, align 4
  %arrayidx9 = getelementptr inbounds float* %c, i32 2
  store float %call3, float* %arrayidx9, align 4
  %arrayidx10 = getelementptr inbounds float* %c, i32 3
  store float %call4, float* %arrayidx10, align 4
  ret void

; CHECK-LABEL: @vec_powi_f32(
; CHECK: load <4 x float>
; CHECK: load <4 x float>
; CHECK: call <4 x float> @llvm.powi.v4f32
; CHECK: store <4 x float>
; CHECK: ret
}


define void @vec_powi_f32_neg(float* %a, float* %b, float* %c, i32 %P, i32 %Q) {
entry:
  %i0 = load float* %a, align 4
  %i1 = load float* %b, align 4
  %add1 = fadd float %i0, %i1
  %call1 = tail call float @llvm.powi.f32(float %add1,i32 %P) nounwind readnone

  %arrayidx2 = getelementptr inbounds float* %a, i32 1
  %i2 = load float* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds float* %b, i32 1
  %i3 = load float* %arrayidx3, align 4
  %add2 = fadd float %i2, %i3
  %call2 = tail call float @llvm.powi.f32(float %add2,i32 %Q) nounwind readnone

  %arrayidx4 = getelementptr inbounds float* %a, i32 2
  %i4 = load float* %arrayidx4, align 4
  %arrayidx5 = getelementptr inbounds float* %b, i32 2
  %i5 = load float* %arrayidx5, align 4
  %add3 = fadd float %i4, %i5
  %call3 = tail call float @llvm.powi.f32(float %add3,i32 %P) nounwind readnone

  %arrayidx6 = getelementptr inbounds float* %a, i32 3
  %i6 = load float* %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds float* %b, i32 3
  %i7 = load float* %arrayidx7, align 4
  %add4 = fadd float %i6, %i7
  %call4 = tail call float @llvm.powi.f32(float %add4,i32 %Q) nounwind readnone

  store float %call1, float* %c, align 4
  %arrayidx8 = getelementptr inbounds float* %c, i32 1
  store float %call2, float* %arrayidx8, align 4
  %arrayidx9 = getelementptr inbounds float* %c, i32 2
  store float %call3, float* %arrayidx9, align 4
  %arrayidx10 = getelementptr inbounds float* %c, i32 3
  store float %call4, float* %arrayidx10, align 4
  ret void

; CHECK-LABEL: @vec_powi_f32_neg(
; CHECK-NOT: call <4 x float> @llvm.powi.v4f32
}
