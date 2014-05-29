; RUN: llc < %s -march=x86 -mcpu=pentium4 -mattr=+sse2 | FileCheck %s -check-prefix=SSE2
; RUN: llc < %s -march=x86 -mcpu=pentium4 -mattr=+sse3 | FileCheck %s -check-prefix=SSE3
; RUN: llc < %s -march=x86-64 -mattr=+avx | FileCheck %s -check-prefix=AVX

define void @test_v4sf(<4 x float>* %P, <4 x float>* %Q, float %X) nounwind {
	%tmp = insertelement <4 x float> zeroinitializer, float %X, i32 0		; <<4 x float>> [#uses=1]
	%tmp2 = insertelement <4 x float> %tmp, float %X, i32 1		; <<4 x float>> [#uses=1]
	%tmp4 = insertelement <4 x float> %tmp2, float %X, i32 2		; <<4 x float>> [#uses=1]
	%tmp6 = insertelement <4 x float> %tmp4, float %X, i32 3		; <<4 x float>> [#uses=1]
	%tmp8 = load <4 x float>* %Q		; <<4 x float>> [#uses=1]
	%tmp10 = fmul <4 x float> %tmp8, %tmp6		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp10, <4 x float>* %P
	ret void

; SSE2-LABEL: test_v4sf:
; SSE2: pshufd $0

; SSE3-LABEL: test_v4sf:
; SSE3: pshufd $0
}

define void @test_v2sd(<2 x double>* %P, <2 x double>* %Q, double %X) nounwind {
	%tmp = insertelement <2 x double> zeroinitializer, double %X, i32 0		; <<2 x double>> [#uses=1]
	%tmp2 = insertelement <2 x double> %tmp, double %X, i32 1		; <<2 x double>> [#uses=1]
	%tmp4 = load <2 x double>* %Q		; <<2 x double>> [#uses=1]
	%tmp6 = fmul <2 x double> %tmp4, %tmp2		; <<2 x double>> [#uses=1]
	store <2 x double> %tmp6, <2 x double>* %P
	ret void

; SSE2-LABEL: test_v2sd:
; SSE2: shufpd $0

; SSE3-LABEL: test_v2sd:
; SSE3: movddup
}

; Fold extract of a load into the load's address computation. This avoids spilling to the stack.
define <4 x float> @load_extract_splat(<4 x float>* nocapture readonly %ptr, i64 %i, i64 %j) nounwind {
  %1 = getelementptr inbounds <4 x float>* %ptr, i64 %i
  %2 = load <4 x float>* %1, align 16
  %3 = trunc i64 %j to i32
  %4 = extractelement <4 x float> %2, i32 %3
  %5 = insertelement <4 x float> undef, float %4, i32 0
  %6 = insertelement <4 x float> %5, float %4, i32 1
  %7 = insertelement <4 x float> %6, float %4, i32 2
  %8 = insertelement <4 x float> %7, float %4, i32 3
  ret <4 x float> %8
  
; AVX-LABEL: load_extract_splat
; AVX-NOT: rsp
; AVX: vbroadcastss
}

; Fold extract of a load into the load's address computation. This avoids spilling to the stack.
define <4 x float> @load_extract_splat1(<4 x float>* nocapture readonly %ptr, i64 %i, i64 %j) nounwind {
  %1 = getelementptr inbounds <4 x float>* %ptr, i64 %i
  %2 = load <4 x float>* %1, align 16
  %3 = extractelement <4 x float> %2, i64 %j
  %4 = insertelement <4 x float> undef, float %3, i32 0
  %5 = insertelement <4 x float> %4, float %3, i32 1
  %6 = insertelement <4 x float> %5, float %3, i32 2
  %7 = insertelement <4 x float> %6, float %3, i32 3
  ret <4 x float> %7
  
; AVX-LABEL: load_extract_splat1
; AVX-NOT: movs
; AVX: vbroadcastss
}
