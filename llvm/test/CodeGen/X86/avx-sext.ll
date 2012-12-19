; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx | FileCheck %s

define <8 x i32> @sext_8i16_to_8i32(<8 x i16> %A) nounwind uwtable readnone ssp {
;CHECK: sext_8i16_to_8i32
;CHECK: vpmovsxwd

  %B = sext <8 x i16> %A to <8 x i32>
  ret <8 x i32>%B
}

define <4 x i64> @sext_4i32_to_4i64(<4 x i32> %A) nounwind uwtable readnone ssp {
;CHECK: sext_4i32_to_4i64
;CHECK: vpmovsxdq

  %B = sext <4 x i32> %A to <4 x i64>
  ret <4 x i64>%B
}

; CHECK: load_sext_test1
; CHECK: vpmovsxwd (%r{{[^,]*}}), %xmm{{.*}}
; CHECK: ret
define <4 x i32> @load_sext_test1(<4 x i16> *%ptr) {
 %X = load <4 x i16>* %ptr
 %Y = sext <4 x i16> %X to <4 x i32>
 ret <4 x i32>%Y
}

; CHECK: load_sext_test2
; CHECK: vpmovsxbd (%r{{[^,]*}}), %xmm{{.*}}
; CHECK: ret
define <4 x i32> @load_sext_test2(<4 x i8> *%ptr) {
 %X = load <4 x i8>* %ptr
 %Y = sext <4 x i8> %X to <4 x i32>
 ret <4 x i32>%Y
}

; CHECK: load_sext_test3
; CHECK: vpmovsxbq (%r{{[^,]*}}), %xmm{{.*}}
; CHECK: ret
define <2 x i64> @load_sext_test3(<2 x i8> *%ptr) {
 %X = load <2 x i8>* %ptr
 %Y = sext <2 x i8> %X to <2 x i64>
 ret <2 x i64>%Y
}

; CHECK: load_sext_test4
; CHECK: vpmovsxwq (%r{{[^,]*}}), %xmm{{.*}}
; CHECK: ret
define <2 x i64> @load_sext_test4(<2 x i16> *%ptr) {
 %X = load <2 x i16>* %ptr
 %Y = sext <2 x i16> %X to <2 x i64>
 ret <2 x i64>%Y
}

; CHECK: load_sext_test5
; CHECK: vpmovsxdq (%r{{[^,]*}}), %xmm{{.*}}
; CHECK: ret
define <2 x i64> @load_sext_test5(<2 x i32> *%ptr) {
 %X = load <2 x i32>* %ptr
 %Y = sext <2 x i32> %X to <2 x i64>
 ret <2 x i64>%Y
}

; CHECK: load_sext_test6
; CHECK: vpmovsxbw (%r{{[^,]*}}), %xmm{{.*}}
; CHECK: ret
define <8 x i16> @load_sext_test6(<8 x i8> *%ptr) {
 %X = load <8 x i8>* %ptr
 %Y = sext <8 x i8> %X to <8 x i16>
 ret <8 x i16>%Y
}
