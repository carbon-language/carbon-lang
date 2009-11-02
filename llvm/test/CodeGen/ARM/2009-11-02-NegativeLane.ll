; RUN: llc -mcpu=cortex-a8 < %s | grep vdup.32
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "armv7-eabi"

define arm_aapcs_vfpcc void @foo(i8* nocapture %pBuffer, i32 %numItems) nounwind {
entry:
  br i1 undef, label %return, label %bb

bb:                                               ; preds = %bb, %entry
  %0 = load float* undef, align 4                 ; <float> [#uses=1]
  %1 = insertelement <4 x float> undef, float %0, i32 2 ; <<4 x float>> [#uses=1]
  %2 = insertelement <4 x float> %1, float undef, i32 3 ; <<4 x float>> [#uses=1]
  %3 = fmul <4 x float> undef, %2                 ; <<4 x float>> [#uses=1]
  %4 = extractelement <4 x float> %3, i32 1       ; <float> [#uses=1]
  store float %4, float* undef, align 4
  br i1 undef, label %return, label %bb

return:                                           ; preds = %bb, %entry
  ret void
}
