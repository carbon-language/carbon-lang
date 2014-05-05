; RUN: opt < %s -basicaa -slp-vectorizer -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; Simple 3-pair chain with loads and stores
; CHECK: test1
define void @test1(double* %a, double* %b, double* %c) {
entry:
  %agg.tmp.i.i.sroa.0 = alloca [3 x double], align 16
; CHECK: %[[V0:[0-9]+]] = load <2 x double>* %[[V2:[0-9]+]], align 8
  %i0 = load double* %a 
  %i1 = load double* %b 
  %mul = fmul double %i0, %i1
  %store1 = getelementptr inbounds [3 x double]* %agg.tmp.i.i.sroa.0, i64 0, i64 1
  %store2 = getelementptr inbounds [3 x double]* %agg.tmp.i.i.sroa.0, i64 0, i64 2
  %arrayidx3 = getelementptr inbounds double* %a, i64 1
  %i3 = load double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double* %b, i64 1
  %i4 = load double* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
; CHECK: store <2 x double> %[[V1:[0-9]+]], <2 x double>* %[[V2:[0-9]+]], align 8
  store double %mul, double* %store1
  store double %mul5, double* %store2, align 16
; CHECK: ret
  ret void
}
