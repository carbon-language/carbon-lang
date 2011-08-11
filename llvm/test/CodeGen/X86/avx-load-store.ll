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

;;
;; The two tests below check that we must fold load + scalar_to_vector
;; + ins_subvec+ zext into only a single vmovss or vmovsd

; CHECK: vmovss (%
define <8 x float> @mov00(<8 x float> %v, float * %ptr) nounwind {
  %val = load float* %ptr
  %i0 = insertelement <8 x float> zeroinitializer, float %val, i32 0
  ret <8 x float> %i0
}

; CHECK: vmovsd (%
define <4 x double> @mov01(<4 x double> %v, double * %ptr) nounwind {
  %val = load double* %ptr
  %i0 = insertelement <4 x double> zeroinitializer, double %val, i32 0
  ret <4 x double> %i0
}

; CHECK: vmovaps  %ymm
define void @storev16i16(<16 x i16> %a) nounwind {
  store <16 x i16> %a, <16 x i16>* undef, align 32
  unreachable
}

; CHECK: vmovups  %ymm
define void @storev16i16_01(<16 x i16> %a) nounwind {
  store <16 x i16> %a, <16 x i16>* undef, align 4
  unreachable
}

; CHECK: vmovaps  %ymm
define void @storev32i8(<32 x i8> %a) nounwind {
  store <32 x i8> %a, <32 x i8>* undef, align 32
  unreachable
}

; CHECK: vmovups  %ymm
define void @storev32i8_01(<32 x i8> %a) nounwind {
  store <32 x i8> %a, <32 x i8>* undef, align 4
  unreachable
}

; It is faster to make two saves, if the data is already in XMM registers. For
; example, after making an integer operation.
; CHECK: _double_save
; CHECK-NOT: vinsertf128 $1
; CHECK-NOT: vinsertf128 $0
; CHECK: vmovaps %xmm
; CHECK: vmovaps %xmm
define void @double_save(<4 x i32> %A, <4 x i32> %B, <8 x i32>* %P) nounwind ssp {
entry:
  %Z = shufflevector <4 x i32>%A, <4 x i32>%B, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x i32> %Z, <8 x i32>* %P, align 16
  ret void
}

