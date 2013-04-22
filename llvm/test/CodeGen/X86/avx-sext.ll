; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx | FileCheck %s -check-prefix=AVX
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core2 | FileCheck %s -check-prefix=SSSE3
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=pentium4 | FileCheck %s -check-prefix=SSE2

define <8 x i32> @sext_8i16_to_8i32(<8 x i16> %A) nounwind uwtable readnone ssp {
; AVX: sext_8i16_to_8i32
; AVX: vpmovsxwd

  %B = sext <8 x i16> %A to <8 x i32>
  ret <8 x i32>%B
}

define <4 x i64> @sext_4i32_to_4i64(<4 x i32> %A) nounwind uwtable readnone ssp {
; AVX: sext_4i32_to_4i64
; AVX: vpmovsxdq

  %B = sext <4 x i32> %A to <4 x i64>
  ret <4 x i64>%B
}

; AVX: load_sext_test1
; AVX: vpmovsxwd (%r{{[^,]*}}), %xmm{{.*}}
; AVX: ret

; SSSE3: load_sext_test1
; SSSE3: movq
; SSSE3: punpcklwd %xmm{{.*}}, %xmm{{.*}}
; SSSE3: psrad $16
; SSSE3: ret

; SSE2: load_sext_test1
; SSE2: movq
; SSE2: punpcklwd %xmm{{.*}}, %xmm{{.*}}
; SSE2: psrad $16
; SSE2: ret
define <4 x i32> @load_sext_test1(<4 x i16> *%ptr) {
 %X = load <4 x i16>* %ptr
 %Y = sext <4 x i16> %X to <4 x i32>
 ret <4 x i32>%Y
}

; AVX: load_sext_test2
; AVX: vpmovsxbd (%r{{[^,]*}}), %xmm{{.*}}
; AVX: ret

; SSSE3: load_sext_test2
; SSSE3: movd
; SSSE3: pshufb
; SSSE3: psrad $24
; SSSE3: ret

; SSE2: load_sext_test2
; SSE2: movl
; SSE2: psrad $24
; SSE2: ret
define <4 x i32> @load_sext_test2(<4 x i8> *%ptr) {
 %X = load <4 x i8>* %ptr
 %Y = sext <4 x i8> %X to <4 x i32>
 ret <4 x i32>%Y
}

; AVX: load_sext_test3
; AVX: vpmovsxbq (%r{{[^,]*}}), %xmm{{.*}}
; AVX: ret

; SSSE3: load_sext_test3
; SSSE3: movsbq
; SSSE3: movsbq
; SSSE3: punpcklqdq
; SSSE3: ret

; SSE2: load_sext_test3
; SSE2: movsbq
; SSE2: movsbq
; SSE2: punpcklqdq
; SSE2: ret
define <2 x i64> @load_sext_test3(<2 x i8> *%ptr) {
 %X = load <2 x i8>* %ptr
 %Y = sext <2 x i8> %X to <2 x i64>
 ret <2 x i64>%Y
}

; AVX: load_sext_test4
; AVX: vpmovsxwq (%r{{[^,]*}}), %xmm{{.*}}
; AVX: ret

; SSSE3: load_sext_test4
; SSSE3: movswq
; SSSE3: movswq
; SSSE3: punpcklqdq
; SSSE3: ret

; SSE2: load_sext_test4
; SSE2: movswq
; SSE2: movswq
; SSE2: punpcklqdq
; SSE2: ret
define <2 x i64> @load_sext_test4(<2 x i16> *%ptr) {
 %X = load <2 x i16>* %ptr
 %Y = sext <2 x i16> %X to <2 x i64>
 ret <2 x i64>%Y
}

; AVX: load_sext_test5
; AVX: vpmovsxdq (%r{{[^,]*}}), %xmm{{.*}}
; AVX: ret

; SSSE3: load_sext_test5
; SSSE3: movslq
; SSSE3: movslq
; SSSE3: punpcklqdq
; SSSE3: ret

; SSE2: load_sext_test5
; SSE2: movslq
; SSE2: movslq
; SSE2: punpcklqdq
; SSE2: ret
define <2 x i64> @load_sext_test5(<2 x i32> *%ptr) {
 %X = load <2 x i32>* %ptr
 %Y = sext <2 x i32> %X to <2 x i64>
 ret <2 x i64>%Y
}

; AVX: load_sext_test6
; AVX: vpmovsxbw (%r{{[^,]*}}), %xmm{{.*}}
; AVX: ret

; SSSE3: load_sext_test6
; SSSE3: movq
; SSSE3: punpcklbw
; SSSE3: psraw $8
; SSSE3: ret

; SSE2: load_sext_test6
; SSE2: movq
; SSE2: punpcklbw
; SSE2: psraw $8
; SSE2: ret
define <8 x i16> @load_sext_test6(<8 x i8> *%ptr) {
 %X = load <8 x i8>* %ptr
 %Y = sext <8 x i8> %X to <8 x i16>
 ret <8 x i16>%Y
}

; AVX: sext_4i1_to_4i64
; AVX: vpslld  $31
; AVX: vpsrad  $31
; AVX: vpmovsxdq
; AVX: vpmovsxdq
; AVX: ret
define <4 x i64> @sext_4i1_to_4i64(<4 x i1> %mask) {
  %extmask = sext <4 x i1> %mask to <4 x i64>
  ret <4 x i64> %extmask
}

; AVX: sext_4i8_to_4i64
; AVX: vpslld  $24
; AVX: vpsrad  $24
; AVX: vpmovsxdq
; AVX: vpmovsxdq
; AVX: ret
define <4 x i64> @sext_4i8_to_4i64(<4 x i8> %mask) {
  %extmask = sext <4 x i8> %mask to <4 x i64>
  ret <4 x i64> %extmask
}

; AVX: sext_4i8_to_4i64
; AVX: vpmovsxbd
; AVX: vpmovsxdq
; AVX: vpmovsxdq
; AVX: ret
define <4 x i64> @load_sext_4i8_to_4i64(<4 x i8> *%ptr) {
 %X = load <4 x i8>* %ptr
 %Y = sext <4 x i8> %X to <4 x i64>
 ret <4 x i64>%Y
}

; AVX: sext_4i16_to_4i64
; AVX: vpmovsxwd
; AVX: vpmovsxdq
; AVX: vpmovsxdq
; AVX: ret
define <4 x i64> @load_sext_4i16_to_4i64(<4 x i16> *%ptr) {
 %X = load <4 x i16>* %ptr
 %Y = sext <4 x i16> %X to <4 x i64>
 ret <4 x i64>%Y
}
