; RUN: opt -S -slp-vectorizer %s -slp-threshold=-10 | FileCheck %s
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

%structA = type { [2 x float] }

define void @test1(%structA* nocapture readonly %J, i32 %xmin, i32 %ymin) {
; CHECK-LABEL: test1
; CHECK: %arrayidx4 = getelementptr inbounds %structA, %structA* %J, i64 0, i32 0, i64 0
; CHECK: %arrayidx9 = getelementptr inbounds %structA, %structA* %J, i64 0, i32 0, i64 1
; CHECK: %3 = bitcast float* %arrayidx4 to <2 x float>*
; CHECK: %4 = load <2 x float>, <2 x float>* %3, align 4
; CHECK: %5 = fsub fast <2 x float> %2, %4
; CHECK: %6 = fmul fast <2 x float> %5, %5
; CHECK: %7 = extractelement <2 x float> %6, i32 0
; CHECK: %8 = extractelement <2 x float> %6, i32 1
; CHECK: %add = fadd fast float %7, %8
; CHECK: %cmp = fcmp oeq float %add, 0.000000e+00

entry:
  br label %for.body3.lr.ph

for.body3.lr.ph:
  %conv5 = sitofp i32 %ymin to float
  %conv = sitofp i32 %xmin to float
  %arrayidx4 = getelementptr inbounds %structA, %structA* %J, i64 0, i32 0, i64 0
  %0 = load float, float* %arrayidx4, align 4
  %sub = fsub fast float %conv, %0
  %arrayidx9 = getelementptr inbounds %structA, %structA* %J, i64 0, i32 0, i64 1
  %1 = load float, float* %arrayidx9, align 4
  %sub10 = fsub fast float %conv5, %1
  %mul11 = fmul fast float %sub, %sub
  %mul12 = fmul fast float %sub10, %sub10
  %add = fadd fast float %mul11, %mul12
  %cmp = fcmp oeq float %add, 0.000000e+00
  br i1 %cmp, label %for.body3.lr.ph, label %for.end27

for.end27:
  ret void
}

define void @test2(%structA* nocapture readonly %J, i32 %xmin, i32 %ymin) {
; CHECK-LABEL: test2
; CHECK: %arrayidx4 = getelementptr inbounds %structA, %structA* %J, i64 0, i32 0, i64 0
; CHECK: %arrayidx9 = getelementptr inbounds %structA, %structA* %J, i64 0, i32 0, i64 1
; CHECK: %3 = bitcast float* %arrayidx4 to <2 x float>*
; CHECK: %4 = load <2 x float>, <2 x float>* %3, align 4
; CHECK: %5 = fsub fast <2 x float> %2, %4
; CHECK: %6 = fmul fast <2 x float> %5, %5
; CHECK: %7 = extractelement <2 x float> %6, i32 0
; CHECK: %8 = extractelement <2 x float> %6, i32 1
; CHECK: %add = fadd fast float %8, %7
; CHECK: %cmp = fcmp oeq float %add, 0.000000e+00

entry:
  br label %for.body3.lr.ph

for.body3.lr.ph:
  %conv5 = sitofp i32 %ymin to float
  %conv = sitofp i32 %xmin to float
  %arrayidx4 = getelementptr inbounds %structA, %structA* %J, i64 0, i32 0, i64 0
  %0 = load float, float* %arrayidx4, align 4
  %sub = fsub fast float %conv, %0
  %arrayidx9 = getelementptr inbounds %structA, %structA* %J, i64 0, i32 0, i64 1
  %1 = load float, float* %arrayidx9, align 4
  %sub10 = fsub fast float %conv5, %1
  %mul11 = fmul fast float %sub, %sub
  %mul12 = fmul fast float %sub10, %sub10
  %add = fadd fast float %mul12, %mul11         ;;;<---- Operands commuted!!
  %cmp = fcmp oeq float %add, 0.000000e+00
  br i1 %cmp, label %for.body3.lr.ph, label %for.end27

for.end27:
  ret void
}
