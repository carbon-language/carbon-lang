; RUN: llc < %s -mtriple=armv7-eabi -mcpu=cortex-a8
; PR7158

define arm_aapcs_vfpcc i32 @main() nounwind {
bb.nph55.bb.nph55.split_crit_edge:
  br label %bb3

bb3:                                              ; preds = %bb3, %bb.nph55.bb.nph55.split_crit_edge
  br i1 undef, label %bb.i19, label %bb3

bb.i19:                                           ; preds = %bb.i19, %bb3
  %0 = insertelement <4 x float> undef, float undef, i32 3 ; <<4 x float>> [#uses=3]
  %1 = fmul <4 x float> %0, %0                    ; <<4 x float>> [#uses=1]
  %2 = bitcast <4 x float> %1 to <2 x double>     ; <<2 x double>> [#uses=0]
  %3 = fmul <4 x float> %0, undef                 ; <<4 x float>> [#uses=0]
  br label %bb.i19
}
