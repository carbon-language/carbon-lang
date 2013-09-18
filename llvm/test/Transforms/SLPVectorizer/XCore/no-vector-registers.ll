; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=xcore  | FileCheck %s

target datalayout = "e-p:32:32:32-a0:0:32-n32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f16:16:32-f32:32:32-f64:32:32"
target triple = "xcore"

; Simple 3-pair chain with loads and stores
; CHECK: test1
; CHECK-NOT: <2 x double>
define void @test1(double* %a, double* %b, double* %c) {
entry:
  %i0 = load double* %a, align 8
  %i1 = load double* %b, align 8
  %mul = fmul double %i0, %i1
  %arrayidx3 = getelementptr inbounds double* %a, i64 1
  %i3 = load double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double* %b, i64 1
  %i4 = load double* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  store double %mul, double* %c, align 8
  %arrayidx5 = getelementptr inbounds double* %c, i64 1
  store double %mul5, double* %arrayidx5, align 8
  ret void
}

