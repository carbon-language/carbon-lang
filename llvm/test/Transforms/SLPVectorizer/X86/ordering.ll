; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

define void @updateModelQPFrame(i32 %m_Bits) {
entry:
  %0 = load double* undef, align 8
  %mul = fmul double undef, %0
  %mul2 = fmul double undef, %mul
  %mul4 = fmul double %0, %mul2
  %mul5 = fmul double undef, 4.000000e+00
  %mul7 = fmul double undef, %mul5
  %conv = sitofp i32 %m_Bits to double
  %mul8 = fmul double %conv, %mul7
  %add = fadd double %mul4, %mul8
  %cmp11 = fcmp olt double %add, 0.000000e+00
  ret void
}
