; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s

declare i32 @llvm.x86.avx512.kortestz(i16, i16) nounwind readnone
; CHECK: test_kortestz
; CHECK: kortestw
; CHECK: sete
define i32 @test_kortestz(i16 %a0, i16 %a1) {
  %res = call i32 @llvm.x86.avx512.kortestz(i16 %a0, i16 %a1) 
  ret i32 %res
}

declare i32 @llvm.x86.avx512.kortestc(i16, i16) nounwind readnone
; CHECK: test_kortestc
; CHECK: kortestw
; CHECK: sbbl
define i32 @test_kortestc(i16 %a0, i16 %a1) {
  %res = call i32 @llvm.x86.avx512.kortestc(i16 %a0, i16 %a1) 
  ret i32 %res
}

define <16 x float> @test_rcp_ps_512(<16 x float> %a0) {
  ; CHECK: vrcp14ps
  %res = call <16 x float> @llvm.x86.avx512.rcp14.ps.512(<16 x float> %a0) ; <<16 x float>> [#uses=1]
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.rcp14.ps.512(<16 x float>) nounwind readnone

define <8 x double> @test_rcp_pd_512(<8 x double> %a0) {
  ; CHECK: vrcp14pd
  %res = call <8 x double> @llvm.x86.avx512.rcp14.pd.512(<8 x double> %a0) ; <<8 x double>> [#uses=1]
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.rcp14.pd.512(<8 x double>) nounwind readnone

define <16 x float> @test_rcp28_ps_512(<16 x float> %a0) {
  ; CHECK: vrcp28ps
  %res = call <16 x float> @llvm.x86.avx512.rcp28.ps.512(<16 x float> %a0) ; <<16 x float>> [#uses=1]
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.rcp28.ps.512(<16 x float>) nounwind readnone

define <8 x double> @test_rcp28_pd_512(<8 x double> %a0) {
  ; CHECK: vrcp28pd
  %res = call <8 x double> @llvm.x86.avx512.rcp28.pd.512(<8 x double> %a0) ; <<8 x double>> [#uses=1]
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.rcp28.pd.512(<8 x double>) nounwind readnone

define <8 x double> @test_rndscale_pd_512(<8 x double> %a0) {
  ; CHECK: vrndscale
  %res = call <8 x double> @llvm.x86.avx512.rndscale.pd.512(<8 x double> %a0, i32 7) ; <<8 x double>> [#uses=1]
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.rndscale.pd.512(<8 x double>, i32) nounwind readnone


define <16 x float> @test_rndscale_ps_512(<16 x float> %a0) {
  ; CHECK: vrndscale
  %res = call <16 x float> @llvm.x86.avx512.rndscale.ps.512(<16 x float> %a0, i32 7) ; <<16 x float>> [#uses=1]
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.rndscale.ps.512(<16 x float>, i32) nounwind readnone


define <16 x float> @test_rsqrt_ps_512(<16 x float> %a0) {
  ; CHECK: vrsqrt14ps
  %res = call <16 x float> @llvm.x86.avx512.rsqrt14.ps.512(<16 x float> %a0) ; <<16 x float>> [#uses=1]
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.rsqrt14.ps.512(<16 x float>) nounwind readnone

define <16 x float> @test_rsqrt28_ps_512(<16 x float> %a0) {
  ; CHECK: vrsqrt28ps
  %res = call <16 x float> @llvm.x86.avx512.rsqrt28.ps.512(<16 x float> %a0) ; <<16 x float>> [#uses=1]
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.rsqrt28.ps.512(<16 x float>) nounwind readnone

define <4 x float> @test_rsqrt14_ss(<4 x float> %a0) {
  ; CHECK: vrsqrt14ss
  %res = call <4 x float> @llvm.x86.avx512.rsqrt14.ss(<4 x float> %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.rsqrt14.ss(<4 x float>) nounwind readnone

define <4 x float> @test_rsqrt28_ss(<4 x float> %a0) {
  ; CHECK: vrsqrt28ss
  %res = call <4 x float> @llvm.x86.avx512.rsqrt28.ss(<4 x float> %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.rsqrt28.ss(<4 x float>) nounwind readnone

define <4 x float> @test_rcp14_ss(<4 x float> %a0) {
  ; CHECK: vrcp14ss
  %res = call <4 x float> @llvm.x86.avx512.rcp14.ss(<4 x float> %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.rcp14.ss(<4 x float>) nounwind readnone

define <4 x float> @test_rcp28_ss(<4 x float> %a0) {
  ; CHECK: vrcp28ss
  %res = call <4 x float> @llvm.x86.avx512.rcp28.ss(<4 x float> %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.rcp28.ss(<4 x float>) nounwind readnone

define <8 x double> @test_sqrt_pd_512(<8 x double> %a0) {
  ; CHECK: vsqrtpd
  %res = call <8 x double> @llvm.x86.avx512.sqrt.pd.512(<8 x double> %a0) ; <<8 x double>> [#uses=1]
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.sqrt.pd.512(<8 x double>) nounwind readnone

define <16 x float> @test_sqrt_ps_512(<16 x float> %a0) {
  ; CHECK: vsqrtps
  %res = call <16 x float> @llvm.x86.avx512.sqrt.ps.512(<16 x float> %a0) ; <<16 x float>> [#uses=1]
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.sqrt.ps.512(<16 x float>) nounwind readnone

define <4 x float> @test_sqrt_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vsqrtssz
  %res = call <4 x float> @llvm.x86.avx512.sqrt.ss(<4 x float> %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.sqrt.ss(<4 x float>, <4 x float>) nounwind readnone

define <2 x double> @test_sqrt_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vsqrtsdz
  %res = call <2 x double> @llvm.x86.avx512.sqrt.sd(<2 x double> %a0, <2 x double> %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.avx512.sqrt.sd(<2 x double>, <2 x double>) nounwind readnone

define i64 @test_x86_sse2_cvtsd2si64(<2 x double> %a0) {
  ; CHECK: vcvtsd2siz
  %res = call i64 @llvm.x86.sse2.cvtsd2si64(<2 x double> %a0) ; <i64> [#uses=1]
  ret i64 %res
}
declare i64 @llvm.x86.sse2.cvtsd2si64(<2 x double>) nounwind readnone

define <2 x double> @test_x86_sse2_cvtsi642sd(<2 x double> %a0, i64 %a1) {
  ; CHECK: vcvtsi2sdqz
  %res = call <2 x double> @llvm.x86.sse2.cvtsi642sd(<2 x double> %a0, i64 %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse2.cvtsi642sd(<2 x double>, i64) nounwind readnone

define <2 x double> @test_x86_avx512_cvtusi642sd(<2 x double> %a0, i64 %a1) {
  ; CHECK: vcvtusi2sdqz
  %res = call <2 x double> @llvm.x86.avx512.cvtusi642sd(<2 x double> %a0, i64 %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.avx512.cvtusi642sd(<2 x double>, i64) nounwind readnone

define i64 @test_x86_sse2_cvttsd2si64(<2 x double> %a0) {
  ; CHECK: vcvttsd2siz
  %res = call i64 @llvm.x86.sse2.cvttsd2si64(<2 x double> %a0) ; <i64> [#uses=1]
  ret i64 %res
}
declare i64 @llvm.x86.sse2.cvttsd2si64(<2 x double>) nounwind readnone


define i64 @test_x86_sse_cvtss2si64(<4 x float> %a0) {
  ; CHECK: vcvtss2siz
  %res = call i64 @llvm.x86.sse.cvtss2si64(<4 x float> %a0) ; <i64> [#uses=1]
  ret i64 %res
}
declare i64 @llvm.x86.sse.cvtss2si64(<4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_cvtsi642ss(<4 x float> %a0, i64 %a1) {
  ; CHECK: vcvtsi2ssqz
  %res = call <4 x float> @llvm.x86.sse.cvtsi642ss(<4 x float> %a0, i64 %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.cvtsi642ss(<4 x float>, i64) nounwind readnone


define i64 @test_x86_sse_cvttss2si64(<4 x float> %a0) {
  ; CHECK: vcvttss2siz
  %res = call i64 @llvm.x86.sse.cvttss2si64(<4 x float> %a0) ; <i64> [#uses=1]
  ret i64 %res
}
declare i64 @llvm.x86.sse.cvttss2si64(<4 x float>) nounwind readnone

define i64 @test_x86_avx512_cvtsd2usi64(<2 x double> %a0) {
  ; CHECK: vcvtsd2usiz
  %res = call i64 @llvm.x86.avx512.cvtsd2usi64(<2 x double> %a0) ; <i64> [#uses=1]
  ret i64 %res
}
declare i64 @llvm.x86.avx512.cvtsd2usi64(<2 x double>) nounwind readnone

define <16 x float> @test_x86_vcvtph2ps_512(<16 x i16> %a0) {
  ; CHECK: vcvtph2ps
  %res = call <16 x float> @llvm.x86.avx512.vcvtph2ps.512(<16 x i16> %a0)
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.vcvtph2ps.512(<16 x i16>) nounwind readonly


define <16 x i16> @test_x86_vcvtps2ph_256(<16 x float> %a0) {
  ; CHECK: vcvtps2ph
  %res = call <16 x i16> @llvm.x86.avx512.vcvtps2ph.512(<16 x float> %a0, i32 0)
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.x86.avx512.vcvtps2ph.512(<16 x float>, i32) nounwind readonly
