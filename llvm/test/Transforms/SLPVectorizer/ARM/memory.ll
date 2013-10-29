; RUN: opt < %s -basicaa -slp-vectorizer -S -mtriple=thumbv7-apple-ios3.0.0 -mcpu=swift | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"

; On swift unaligned <2 x double> stores need 4uops and it is there for cheaper
; to do this scalar.

; CHECK-LABEL: expensive_double_store
; CHECK-NOT: load <2 x double>
; CHECK-NOT: store <2 x double>
define void @expensive_double_store(double* noalias %dst, double* noalias %src, i64 %count) {
entry:
  %0 = load double* %src, align 8
  store double %0, double* %dst, align 8
  %arrayidx2 = getelementptr inbounds double* %src, i64 1
  %1 = load double* %arrayidx2, align 8
  %arrayidx3 = getelementptr inbounds double* %dst, i64 1
  store double %1, double* %arrayidx3, align 8
  ret void
}
