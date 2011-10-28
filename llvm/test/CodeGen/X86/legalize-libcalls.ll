; RUN: llc -march=x86 < %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.7.0"

define float @MakeSphere(float %theta.079) nounwind {
entry:
  %add36 = fadd float %theta.079, undef
  %call = call float @cosf(float %theta.079) nounwind readnone
  %call45 = call float @sinf(float %theta.079) nounwind readnone
  %call37 = call float @sinf(float %add36) nounwind readnone
  store float %call, float* undef, align 8
  store float %call37, float* undef, align 8
  store float %call45, float* undef, align 8
  ret float %add36
}

declare float @cosf(float) nounwind readnone
declare float @sinf(float) nounwind readnone
