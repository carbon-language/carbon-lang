; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: vmovaps
; CHECK: vmovaps
; CHECK: vmovapd
; CHECK: vmovapd
; CHECK: vmovaps
; CHECK: vmovaps
define void @test_256_load(double* nocapture %d, float* nocapture %f, <4 x i64>* nocapture %i) nounwind uwtable ssp {
entry:
  %0 = bitcast double* %d to <4 x double>*
  %tmp1.i = load <4 x double>* %0, align 32
  %1 = bitcast float* %f to <8 x float>*
  %tmp1.i17 = load <8 x float>* %1, align 32
  %tmp1.i16 = load <4 x i64>* %i, align 32
  tail call void @dummy(<4 x double> %tmp1.i, <8 x float> %tmp1.i17, <4 x i64> %tmp1.i16) nounwind
  store <4 x double> %tmp1.i, <4 x double>* %0, align 32
  store <8 x float> %tmp1.i17, <8 x float>* %1, align 32
  store <4 x i64> %tmp1.i16, <4 x i64>* %i, align 32
  ret void
}

declare void @dummy(<4 x double>, <8 x float>, <4 x i64>)

