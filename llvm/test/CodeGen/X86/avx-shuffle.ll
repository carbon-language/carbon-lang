; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; PR11102
define <4 x float> @test1(<4 x float> %a) nounwind {
  %b = shufflevector <4 x float> zeroinitializer, <4 x float> %a, <4 x i32> <i32 2, i32 5, i32 undef, i32 undef>
  ret <4 x float> %b
; CHECK-LABEL: test1:
; CHECK: vshufps
; CHECK: vpshufd
}

; rdar://10538417
define <3 x i64> @test2(<2 x i64> %v) nounwind readnone {
; CHECK-LABEL: test2:
; CHECK: vinsertf128
  %1 = shufflevector <2 x i64> %v, <2 x i64> %v, <3 x i32> <i32 0, i32 1, i32 undef>
  %2 = shufflevector <3 x i64> zeroinitializer, <3 x i64> %1, <3 x i32> <i32 3, i32 4, i32 2>
  ret <3 x i64> %2
; CHECK: ret
}

define <4 x i64> @test3(<4 x i64> %a, <4 x i64> %b) nounwind {
  %c = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 4, i32 5, i32 2, i32 undef>
  ret <4 x i64> %c
; CHECK-LABEL: test3:
; CHECK: vperm2f128
; CHECK: ret
}

define <8 x float> @test4(float %a) nounwind {
  %b = insertelement <8 x float> zeroinitializer, float %a, i32 0
  ret <8 x float> %b
; CHECK-LABEL: test4:
; CHECK: vinsertf128
}

; rdar://10594409
define <8 x float> @test5(float* nocapture %f) nounwind uwtable readonly ssp {
entry:
  %0 = bitcast float* %f to <4 x float>*
  %1 = load <4 x float>* %0, align 16
; CHECK: test5
; CHECK: vmovaps
; CHECK-NOT: vxorps
; CHECK-NOT: vinsertf128
  %shuffle.i = shufflevector <4 x float> %1, <4 x float> <float 0.000000e+00, float undef, float undef, float undef>, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 4, i32 4, i32 4>
  ret <8 x float> %shuffle.i
}

define <4 x double> @test6(double* nocapture %d) nounwind uwtable readonly ssp {
entry:
  %0 = bitcast double* %d to <2 x double>*
  %1 = load <2 x double>* %0, align 16
; CHECK: test6
; CHECK: vmovaps
; CHECK-NOT: vxorps
; CHECK-NOT: vinsertf128
  %shuffle.i = shufflevector <2 x double> %1, <2 x double> <double 0.000000e+00, double undef>, <4 x i32> <i32 0, i32 1, i32 2, i32 2>
  ret <4 x double> %shuffle.i
}

define <16 x i16> @test7(<4 x i16> %a) nounwind {
; CHECK: test7
  %b = shufflevector <4 x i16> %a, <4 x i16> undef, <16 x i32> <i32 1, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
; CHECK: ret
  ret <16 x i16> %b
}

; CHECK: test8
define void @test8() {
entry:
  %0 = load <16 x i64> addrspace(1)* null, align 128
  %1 = shufflevector <16 x i64> <i64 undef, i64 undef, i64 0, i64 undef, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 undef, i64 0, i64 undef, i64 undef, i64 undef, i64 undef>, <16 x i64> %0, <16 x i32> <i32 17, i32 18, i32 2, i32 undef, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 undef, i32 11, i32 undef, i32 undef, i32 undef, i32 26>
  %2 = shufflevector <16 x i64> %1, <16 x i64> %0, <16 x i32> <i32 0, i32 1, i32 2, i32 30, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 undef, i32 11, i32 undef, i32 22, i32 20, i32 15>
  store <16 x i64> %2, <16 x i64> addrspace(1)* undef, align 128
; CHECK: ret
  ret void
}

; Extract a value from a shufflevector..
define i32 @test9(<4 x i32> %a) nounwind {
; CHECK: test9
; CHECK: vpextrd
  %b = shufflevector <4 x i32> %a, <4 x i32> undef, <8 x i32> <i32 1, i32 1, i32 2, i32 2, i32 3, i32 3, i32 undef, i32 4>
  %r = extractelement <8 x i32> %b, i32 2
; CHECK: ret
  ret i32 %r
}

; Extract a value which is the result of an undef mask.
define i32 @test10(<4 x i32> %a) nounwind {
; CHECK: @test10
; CHECK-NOT: {{^[^#]*[a-z]}}
; CHECK: ret
  %b = shufflevector <4 x i32> %a, <4 x i32> undef, <8 x i32> <i32 1, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %r = extractelement <8 x i32> %b, i32 2
  ret i32 %r
}

define <4 x float> @test11(<4 x float> %a) nounwind  {
; CHECK: test11
; CHECK: vpshufd $27
  %tmp1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x float> %tmp1
}

define <4 x float> @test12(<4 x float>* %a) nounwind  {
; CHECK: test12
; CHECK: vpshufd
  %tmp0 = load <4 x float>* %a
  %tmp1 = shufflevector <4 x float> %tmp0, <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x float> %tmp1
}

define <4 x i32> @test13(<4 x i32> %a) nounwind  {
; CHECK: test13
; CHECK: vpshufd $27
  %tmp1 = shufflevector <4 x i32> %a, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x i32> %tmp1
}

define <4 x i32> @test14(<4 x i32>* %a) nounwind  {
; CHECK: test14
; CHECK: vpshufd $27, (
  %tmp0 = load <4 x i32>* %a
  %tmp1 = shufflevector <4 x i32> %tmp0, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x i32> %tmp1
}

; CHECK: test15
; CHECK: vpshufd $8
; CHECK: ret
define <4 x i32> @test15(<2 x i32>%x) nounwind readnone {
  %x1 = shufflevector <2 x i32> %x, <2 x i32> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  ret <4 x i32>%x1
}

; rdar://10974078
define <8 x float> @test16(float* nocapture %f) nounwind uwtable readonly ssp {
entry:
  %0 = bitcast float* %f to <4 x float>*
  %1 = load <4 x float>* %0, align 8
; CHECK: test16
; CHECK: vmovups
; CHECK-NOT: vxorps
; CHECK-NOT: vinsertf128
  %shuffle.i = shufflevector <4 x float> %1, <4 x float> <float 0.000000e+00, float undef, float undef, float undef>, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 4, i32 4, i32 4>
  ret <8 x float> %shuffle.i
}

; PR12413
; CHECK: shuf1
; CHECK: vpshufb
; CHECK: vpshufb
; CHECK: vpshufb
; CHECK: vpshufb
define <32 x i8> @shuf1(<32 x i8> %inval1, <32 x i8> %inval2) {
entry:
 %0 = shufflevector <32 x i8> %inval1, <32 x i8> %inval2, <32 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30, i32 32, i32 34, i32 36, i32 38, i32 40, i32 42, i32 44, i32 46, i32 48, i32 50, i32 52, i32 54, i32 56, i32 58, i32 60, i32 62>
 ret <32 x i8> %0
}

; handle the case where only half of the 256-bits is splittable
; CHECK: shuf2
; CHECK: vpshufb
; CHECK: vpshufb
; CHECK: vpextrb
; CHECK: vpextrb
define <32 x i8> @shuf2(<32 x i8> %inval1, <32 x i8> %inval2) {
entry:
 %0 = shufflevector <32 x i8> %inval1, <32 x i8> %inval2, <32 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30, i32 31, i32 34, i32 36, i32 38, i32 40, i32 42, i32 44, i32 46, i32 48, i32 50, i32 52, i32 54, i32 56, i32 58, i32 60, i32 62>
 ret <32 x i8> %0
}

; CHECK: blend1
; CHECK: vblendps
; CHECK: ret
define <4 x i32> @blend1(<4 x i32> %a, <4 x i32> %b) nounwind alwaysinline {
  %t = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  ret <4 x i32> %t
}

; CHECK: blend2
; CHECK: vblendps
; CHECK: ret
define <4 x i32> @blend2(<4 x i32> %a, <4 x i32> %b) nounwind alwaysinline {
  %t = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x i32> %t
}

; CHECK: blend2a
; CHECK: vblendps
; CHECK: ret
define <4 x float> @blend2a(<4 x float> %a, <4 x float> %b) nounwind alwaysinline {
  %t = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x float> %t
}

; CHECK: blend3
; CHECK-NOT: vblendps
; CHECK: ret
define <4 x i32> @blend3(<4 x i32> %a, <4 x i32> %b) nounwind alwaysinline {
  %t = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 5, i32 2, i32 7>
  ret <4 x i32> %t
}

; CHECK: blend4
; CHECK: vblendpd
; CHECK: ret
define <4 x i64> @blend4(<4 x i64> %a, <4 x i64> %b) nounwind alwaysinline {
  %t = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  ret <4 x i64> %t
}

; CHECK: narrow
; CHECK: vpermilps
; CHECK: ret
define <16 x i16> @narrow(<16 x i16> %a) nounwind alwaysinline {
  %t = shufflevector <16 x i16> %a, <16 x i16> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 1, i32 6, i32 7, i32 4, i32 5, i32 10, i32 11, i32 8, i32 undef, i32 14, i32 15, i32 undef, i32 undef>
  ret <16 x i16> %t
}

;CHECK-LABEL: test17:
;CHECK-NOT: vinsertf128
;CHECK: ret
define   <8 x float> @test17(<4 x float> %y) {
  %x = shufflevector <4 x float> %y, <4 x float> undef, <8 x i32> <i32 undef, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x float> %x
}

; CHECK: test18
; CHECK: vmovshdup
; CHECK: vblendps
; CHECK: ret
define <8 x float> @test18(<8 x float> %A, <8 x float>%B) nounwind {
  %S = shufflevector <8 x float> %A, <8 x float> %B, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  ret <8 x float>%S
}

; CHECK: test19
; CHECK: vmovsldup
; CHECK: vblendps
; CHECK: ret
define <8 x float> @test19(<8 x float> %A, <8 x float>%B) nounwind {
  %S = shufflevector <8 x float> %A, <8 x float> %B, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  ret <8 x float>%S
}

; rdar://12684358
; Make sure loads happen before stores.
; CHECK: swap8doubles
; CHECK: vmovups {{[0-9]*}}(%rdi), %xmm{{[0-9]+}}
; CHECK: vmovups {{[0-9]*}}(%rdi), %xmm{{[0-9]+}}
; CHECK: vinsertf128 $1, {{[0-9]*}}(%rdi), %ymm{{[0-9]+}}
; CHECK: vinsertf128 $1, {{[0-9]*}}(%rdi), %ymm{{[0-9]+}}
; CHECK: vmovaps {{[0-9]*}}(%rsi), %ymm{{[0-9]+}}
; CHECK: vmovaps {{[0-9]*}}(%rsi), %ymm{{[0-9]+}}
; CHECK: vmovaps %xmm{{[0-9]+}}, {{[0-9]*}}(%rdi)
; CHECK: vextractf128
; CHECK: vmovaps %xmm{{[0-9]+}}, {{[0-9]*}}(%rdi)
; CHECK: vextractf128
; CHECK: vmovaps %ymm{{[0-9]+}}, {{[0-9]*}}(%rsi)
; CHECK: vmovaps %ymm{{[0-9]+}}, {{[0-9]*}}(%rsi)
define void @swap8doubles(double* nocapture %A, double* nocapture %C) nounwind uwtable ssp {
entry:
  %add.ptr = getelementptr inbounds double* %A, i64 2
  %v.i = bitcast double* %A to <2 x double>*
  %0 = load <2 x double>* %v.i, align 1
  %shuffle.i.i = shufflevector <2 x double> %0, <2 x double> <double 0.000000e+00, double undef>, <4 x i32> <i32 0, i32 1, i32 2, i32 2>
  %v1.i = bitcast double* %add.ptr to <2 x double>*
  %1 = load <2 x double>* %v1.i, align 1
  %2 = tail call <4 x double> @llvm.x86.avx.vinsertf128.pd.256(<4 x double> %shuffle.i.i, <2 x double> %1, i8 1) nounwind
  %add.ptr1 = getelementptr inbounds double* %A, i64 6
  %add.ptr2 = getelementptr inbounds double* %A, i64 4
  %v.i27 = bitcast double* %add.ptr2 to <2 x double>*
  %3 = load <2 x double>* %v.i27, align 1
  %shuffle.i.i28 = shufflevector <2 x double> %3, <2 x double> <double 0.000000e+00, double undef>, <4 x i32> <i32 0, i32 1, i32 2, i32 2>
  %v1.i29 = bitcast double* %add.ptr1 to <2 x double>*
  %4 = load <2 x double>* %v1.i29, align 1
  %5 = tail call <4 x double> @llvm.x86.avx.vinsertf128.pd.256(<4 x double> %shuffle.i.i28, <2 x double> %4, i8 1) nounwind
  %6 = bitcast double* %C to <4 x double>*
  %7 = load <4 x double>* %6, align 32
  %add.ptr5 = getelementptr inbounds double* %C, i64 4
  %8 = bitcast double* %add.ptr5 to <4 x double>*
  %9 = load <4 x double>* %8, align 32
  %shuffle.i26 = shufflevector <4 x double> %7, <4 x double> undef, <2 x i32> <i32 0, i32 1>
  %10 = tail call <2 x double> @llvm.x86.avx.vextractf128.pd.256(<4 x double> %7, i8 1)
  %shuffle.i = shufflevector <4 x double> %9, <4 x double> undef, <2 x i32> <i32 0, i32 1>
  %11 = tail call <2 x double> @llvm.x86.avx.vextractf128.pd.256(<4 x double> %9, i8 1)
  store <2 x double> %shuffle.i26, <2 x double>* %v.i, align 16
  store <2 x double> %10, <2 x double>* %v1.i, align 16
  store <2 x double> %shuffle.i, <2 x double>* %v.i27, align 16
  store <2 x double> %11, <2 x double>* %v1.i29, align 16
  store <4 x double> %2, <4 x double>* %6, align 32
  store <4 x double> %5, <4 x double>* %8, align 32
  ret void
}
declare <2 x double> @llvm.x86.avx.vextractf128.pd.256(<4 x double>, i8) nounwind readnone
declare <4 x double> @llvm.x86.avx.vinsertf128.pd.256(<4 x double>, <2 x double>, i8) nounwind readnone

; this test case just should not fail
define void @test20() {
  %a0 = insertelement <3 x double> <double 0.000000e+00, double 0.000000e+00, double undef>, double 0.000000e+00, i32 2
  store <3 x double> %a0, <3 x double>* undef, align 1
  %a1 = insertelement <3 x double> <double 0.000000e+00, double 0.000000e+00, double undef>, double undef, i32 2
  store <3 x double> %a1, <3 x double>* undef, align 1
  ret void
}

define <2 x i64> @test_insert_64_zext(<2 x i64> %i) {
; CHECK-LABEL: test_insert_64_zext
; CHECK-NOT: xor
; CHECK: vmovq
  %1 = shufflevector <2 x i64> %i, <2 x i64> <i64 0, i64 undef>, <2 x i32> <i32 0, i32 2>
  ret <2 x i64> %1
}

;; Ensure we don't use insertps from non v4x32 vectors.
;; On SSE4.1 it works because bigger vectors use more than 1 register.
;; On AVX they get passed in a single register.
;; FIXME: We could probably optimize this case, if we're only using the
;; first 4 indices.
define <4 x i32> @insert_from_diff_size(<8 x i32> %x) {
; CHECK-LABEL: insert_from_diff_size:
; CHECK-NOT: insertps
; CHECK: ret
  %vecext = extractelement <8 x i32> %x, i32 0
  %vecinit = insertelement <4 x i32> undef, i32 %vecext, i32 0
  %vecinit1 = insertelement <4 x i32> %vecinit, i32 0, i32 1
  %vecinit2 = insertelement <4 x i32> %vecinit1, i32 0, i32 2
  %a.0 = extractelement <8 x i32> %x, i32 0
  %vecinit3 = insertelement <4 x i32> %vecinit2, i32 %a.0, i32 3
  ret <4 x i32> %vecinit3
}
